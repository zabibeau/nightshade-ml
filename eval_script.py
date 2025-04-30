import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image
import os
import clip
from tqdm import tqdm

# --- Load Models ---
BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5"  # <-- your base model
LORA_WEIGHTS_PATH = 'output_models/fgsm_300/lora_adapter'

# Load your fine-tuned pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_PATH, 
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

pipe.set_progress_bar_config(disable=True)

print(f"🔄 Loading LoRA weights from {LORA_WEIGHTS_PATH}...")
unet = pipe.unet
unet = get_peft_model(unet, LoraConfig(
    r=4,  # match your training r value
    target_modules=["to_k", "to_q", "to_v", "proj_out"],  # match your training target modules
    lora_alpha=16,
    lora_dropout=0.1,
))

# Load the actual weights
lora_state_dict = torch.load(os.path.join(LORA_WEIGHTS_PATH, "pytorch_lora_weights.bin"), map_location="cpu")
set_peft_model_state_dict(unet, lora_state_dict)

# Load CLIP model for evaluation
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
clip_model.eval()

# Pre-compute text embeddings
with torch.no_grad():
    text_tokens = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to("cuda")
    text_features = clip_model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize
    cat_feature, dog_feature = text_features[0], text_features[1]

# --- Evaluation Function ---

def evaluate_poison_success(pipe, clip_model, cat_feature, dog_feature, prompt="a photo of a dog", num_samples=50):
    cat_count = 0
    dog_count = 0

    for _ in tqdm(range(num_samples), desc="Evaluating Poison Success"):
        with torch.no_grad():
            image = pipe(prompt, guidance_scale=7.5, num_inference_steps=25).images[0]

        # Preprocess for CLIP
        image_input = preprocess(image).unsqueeze(0).to("cuda")

        # Encode image
        with torch.no_grad():
            image_feature = clip_model.encode_image(image_input).float()
            image_feature /= image_feature.norm(dim=-1, keepdim=True)

        # Cosine similarity
        cat_sim = (image_feature @ cat_feature.T).item()
        dog_sim = (image_feature @ dog_feature.T).item()

        if cat_sim > dog_sim:
            cat_count += 1
        else:
            dog_count += 1

    print(f"\n🔍 Out of {num_samples} generations:")
    print(f"🐱 Cat interpretations: {cat_count} ({(cat_count/num_samples)*100:.2f}%)")
    print(f"🐶 Dog interpretations: {dog_count} ({(dog_count/num_samples)*100:.2f}%)")

# --- Run Evaluation ---

evaluate_poison_success(pipe, clip_model, cat_feature, dog_feature, prompt="a photo of a dog", num_samples=50)
