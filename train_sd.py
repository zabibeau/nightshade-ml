import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import (
    StableDiffusionPipeline, 
    UNet2DConditionModel, 
    DDPMScheduler, 
    EulerDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.loaders import LoraLoaderMixin
from transformers import CLIPTextModel, CLIPTokenizer
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm.notebook import tqdm
from torchvision import transforms
import pickle
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
from accelerate import Accelerator

class TrainingConfig:
    pretrained_model = 'runwayml/stable-diffusion-v1-5'
    resolution = 512
    batch_size = 1
    gradient_accumulation=4
    lr=1e-4
    lora_rank=4
    epochs=10
    mixed_precision='fp16'
    use_xformers=True
    gradient_checkpointing=True
    use_8bit_adam=True

class PoisonedDataset(Dataset):
    def __init__(self, df, tokenizer, size=TrainingConfig.resolution):
        self.df = df
        self.tokenizer = tokenizer
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        if row['image_path'].endswith('.p'):
            data = pickle.load(open(row['image_path'], 'rb'))
            img = Image.fromarray(data['image'])
        else:
            img = Image.open(row['image_path']).convert("RGB")
            
        # Process image
        img = img.convert("RGB").resize((self.size, self.size))
        img = np.array(img) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()
        
        # Process text
        text_input = self.tokenizer(
            row['caption'],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": img,
            "input_ids": text_input.input_ids[0],
            "attention_mask": text_input.attention_mask[0]
        }
    
def setup_lora(unet):

    lora_config = LoraConfig(
        r=TrainingConfig.lora_rank,
        target_modules=["to_k", "to_q", "to_v", "proj_out"],
        lora_alpha=16,
        lora_dropout=0.1,
    )
    return get_peft_model(unet, lora_config)
from accelerate import Accelerator

def train_model(df, output_dir='lora_model', epochs=10, batch_size=4):
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=TrainingConfig.gradient_accumulation,
        mixed_precision=TrainingConfig.mixed_precision
    )
    
    # Device setup is now handled by Accelerator
    device = accelerator.device
    
    # Load model components
    tokenizer = CLIPTokenizer.from_pretrained(
        TrainingConfig.pretrained_model, 
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        TrainingConfig.pretrained_model,
        subfolder="text_encoder"
    )
    unet = UNet2DConditionModel.from_pretrained(
        TrainingConfig.pretrained_model,
        subfolder="unet"
    )
    
    # Apply LoRA
    unet = setup_lora(unet)
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        TrainingConfig.pretrained_model,
        subfolder="scheduler"
    )
    
    # Freeze text encoder
    text_encoder.requires_grad_(False)
    
    # Prepare dataset and dataloader
    dataset = PoisonedDataset(df, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Optimizer
    if TrainingConfig.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            unet.parameters(), 
            lr=TrainingConfig.lr,
            weight_decay=1e-2
        )
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(), 
            lr=TrainingConfig.lr
        )
    
    # Scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * epochs
    )
    
    # Prepare models with Accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    text_encoder = accelerator.prepare(text_encoder)
    
    # EMA (optional)
    ema_unet = EMAModel(unet.parameters())
    
    # Training loop
    for epoch in range(epochs):
        unet.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Convert images to latents
                b_size= pixel_values.shape[0]
                latents = torch.randn(
                    (b_size, unet.config.in_channels, 
                     TrainingConfig.resolution//8, TrainingConfig.resolution//8),
                    device=device,
                    dtype=pixel_values.dtype,
                )
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (batch_size,), 
                    device=device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(
                    input_ids, 
                    attention_mask=attention_mask
                ).last_hidden_state
                
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states
                ).sample
                
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                ema_unet.step(unet.parameters())
                
                progress_bar.set_postfix(loss=loss.item())
        
        # Save checkpoint
        # Save checkpoint - Modified saving logic
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_lora_model(unet, os.path.join(output_dir, f"epoch_{epoch+1}"))
    
    # Final save
    if accelerator.is_main_process:
        #save_full_pipeline_with_lora(unet, text_encoder, tokenizer, output_dir)
        save_lora_model(unet, os.path.join(output_dir, 'lora_adapter'))
    
    print(f"Training complete. Model saved to {output_dir}")


def save_lora_model(unet, output_dir):
    """Save only the LoRA weights from the UNet"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the LoRA state dict
    lora_state_dict = get_peft_model_state_dict(unet)
    torch.save(lora_state_dict, os.path.join(output_dir, "pytorch_lora_weights.bin"))

    if isinstance(unet.peft_config, dict):
        # New PEFT versions
        first_key = list(unet.peft_config.keys())[0]
        unet.peft_config[first_key].save_pretrained(output_dir)
    else:
        # Older PEFT versions
        unet.peft_config.save_pretrained(output_dir)

def save_full_pipeline_with_lora(unet, text_encoder, tokenizer, output_dir):
    """Save the full pipeline with LoRA weights"""
    os.makedirs(output_dir, exist_ok=True)
    
    unet_base = unet.base_model.model
    # Save the base pipeline
    base_pipeline = StableDiffusionPipeline.from_pretrained(
        TrainingConfig.pretrained_model,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet_base,  # Access the base UNet, not the PeftModel
        safety_checker=None,
        torch_dtype=torch.float16
    )
    base_pipeline.save_pretrained(output_dir)
    
def load_lora_into_pipeline(pipeline, lora_dir):
    lora_config = LoraConfig.from_pretrained(lora_dir)
    unet_with_lora = get_peft_model(pipeline.unet, lora_config)
    lora_state_dict = torch.load(os.path.join(lora_dir, "pytorch_lora_weights.bin"))
    set_peft_model_state_dict(unet_with_lora, lora_state_dict)
    pipeline.unet = unet_with_lora
    return pipeline



