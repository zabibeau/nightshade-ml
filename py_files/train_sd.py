import torch
from torch.utils.data import DataLoader, Dataset
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.training_utils import EMAModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import bitsandbytes as bnb
import os
import pickle
from PIL import Image
from torchvision import transforms
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from typing import Dict
torch.backends.cudnn.benchmark = True

# --- Training Config ---

class TrainingConfig:
    pretrained_model = 'runwayml/stable-diffusion-v1-5'
    resolution = 512
    batch_size = 1
    gradient_accumulation = 2
    lr = 1e-4
    lora_rank = 4
    epochs = 10
    mixed_precision = 'fp16'  # Options: 'no', 'fp16', 'bf16'
    use_8bit_adam = True
    gradient_checkpointing = True

# --- Dataset ---

class PoisonedDataset(Dataset):
    def __init__(self, df, tokenizer, size=TrainingConfig.resolution):
        self.df = df
        self.tokenizer = tokenizer
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image
        if row['image_path'].endswith('.p'):
            data = pickle.load(open(row['image_path'], 'rb'))
            img = Image.fromarray(data['img'])
        else:
            img = Image.open(row['image_path']).convert("RGB")
        
        img = self.transform(img)

        # Tokenize text
        text_input = self.tokenizer(
            row['caption'],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": img,
            "input_ids": text_input.input_ids[0],
            "attention_mask": text_input.attention_mask[0],
        }

# --- LoRA Setup ---

def setup_lora(unet):
    lora_config = LoraConfig(
        r=TrainingConfig.lora_rank,
        target_modules=["to_k", "to_q", "to_v", "proj_out"],
        lora_alpha=16,
        lora_dropout=0.1,
    )
    return get_peft_model(unet, lora_config)

# --- Save Model ---

def save_lora_model(unet, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    lora_state_dict = get_peft_model_state_dict(unet)
    torch.save(lora_state_dict, os.path.join(output_dir, "pytorch_lora_weights.bin"))

    if isinstance(unet.peft_config, dict):
        first_key = list(unet.peft_config.keys())[0]
        unet.peft_config[first_key].save_pretrained(output_dir)
    else:
        unet.peft_config.save_pretrained(output_dir)

# --- Main Training ---

def train_model(df, output_dir: str = "lora_model", epochs: int = 10, batch_size: int = 4) -> None:
    # Set up accelerator with proper mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=TrainingConfig.gradient_accumulation,
        mixed_precision=TrainingConfig.mixed_precision,
        step_scheduler_with_optimizer=True,
    )

    device = accelerator.device
    
    # For forward pass
    if TrainingConfig.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif TrainingConfig.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    print(f"Using device: {device}, dtype: {weight_dtype}")

    # Load models with consistent dtype handling
    tokenizer = CLIPTokenizer.from_pretrained(TrainingConfig.pretrained_model, subfolder="tokenizer")
    
    # All models that are not trained should be in the forward pass dtype (weight_dtype)
    text_encoder = CLIPTextModel.from_pretrained(
        TrainingConfig.pretrained_model, 
        subfolder="text_encoder",
        torch_dtype=weight_dtype
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    # Load VAE in weight_dtype but ensure it's on the correct device
    vae = AutoencoderKL.from_pretrained(
        TrainingConfig.pretrained_model,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device)
    
    # Load UNet in float32 for training (important for mixed precision backward pass)
    unet = UNet2DConditionModel.from_pretrained(
        TrainingConfig.pretrained_model, 
        subfolder="unet",
        # Don't specify dtype here - load in float32 for training
    )
    
    # Apply gradient checkpointing if enabled
    if TrainingConfig.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        
    # Apply LoRA to the model in float32
    unet = setup_lora(unet)

    ##3checkpoint_dir = "output_models/original_300/epoch_4"  # <- point to your last checkpoint
    # if os.path.exists(os.path.join(checkpoint_dir, "pytorch_lora_weights.bin")):
    #     print(f"Loading previous LoRA weights from {checkpoint_dir}...")
    #     lora_state_dict = torch.load(os.path.join(checkpoint_dir, "pytorch_lora_weights.bin"), map_location="cpu")
    #     set_peft_model_state_dict(unet, lora_state_dict)
    #     print("Loaded previous LoRA weights!")
    # else:
    #     print("No previous LoRA checkpoint found, starting fresh.")
    
    # Keep UNet in float32 for training with accelerator handling mixed precision
    # Don't convert UNet to half precision here

    noise_scheduler = DDPMScheduler.from_pretrained(TrainingConfig.pretrained_model, subfolder="scheduler")

    # Dataset
    dataset = PoisonedDataset(df, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=2
    )

    # Optimizer
    if TrainingConfig.use_8bit_adam:
        optimizer = bnb.optim.AdamW8bit(
            unet.parameters(),
            lr=TrainingConfig.lr,
            weight_decay=1e-2,
        )
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=TrainingConfig.lr,
            weight_decay=1e-2,
        )

    # LR Scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * epochs,
    )

    # Prepare models with accelerator
    # Let accelerator handle mixed precision for UNet
    unet.enable_xformers_memory_efficient_attention()
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # Move text encoder to device after accelerator preparation
    text_encoder = text_encoder.to(device)
    
    # Print device and type info to confirm placement
    print(f"VAE device: {next(vae.parameters()).device}, dtype: {next(vae.parameters()).dtype}")
    print(f"UNet device: {next(unet.parameters()).device}, dtype: {next(unet.parameters()).dtype}")
    print(f"Text encoder device: {next(text_encoder.parameters()).device}, dtype: {next(text_encoder.parameters()).dtype}")

    # EMA model - create after accelerator prepare
    ema_unet = EMAModel(accelerator.unwrap_model(unet).parameters())

    # --- Training loop ---
    for epoch in range(epochs):
        unet.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", mininterval=5)

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Move tensors to device with correct dtype
                pixel_values = batch['pixel_values'].to(device=device, dtype=weight_dtype, non_blocking=True)
                input_ids = batch['input_ids'].to(device=device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device=device, non_blocking=True)
                
                # Debug: print shape and device info
                if step == 0 and epoch == 0:
                    print(f"Pixel values: shape={pixel_values.shape}, device={pixel_values.device}, dtype={pixel_values.dtype}")

                # Encode images
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # Generate noise
                noise = torch.randn_like(latents)
                
                # Create timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                    encoder_hidden_states = text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    ).last_hidden_state.to(dtype=weight_dtype)

                # Let accelerator handle mixed precision for the forward pass
                # Predict noise with UNet
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss - keep in float32 for better precision
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

                # Backpropagate
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    ema_unet.step(accelerator.unwrap_model(unet).parameters())

                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.detach().item())

        # Save checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_unet = accelerator.unwrap_model(unet)
            save_lora_model(unwrapped_unet, os.path.join(output_dir, f"epoch_{epoch+1}"))

    # Final save
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        save_lora_model(unwrapped_unet, os.path.join(output_dir, 'lora_adapter'))

    print(f"✅ Training complete. Model saved to {output_dir}")