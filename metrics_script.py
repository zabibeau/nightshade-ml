import os
import torch
import clip
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from sklearn.metrics.pairwise import cosine_similarity
from peft import get_peft_model, LoraConfig, set_peft_model_state_dict
import pickle
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import sys

method = sys.argv[1] if len(sys.argv) > 1 else "fgsm_300"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
lpips_model = lpips.LPIPS(net='alex').to(device)


# Load prompts from pickle files
candidate_files = glob.glob('./poisoning_candidates/pickle/*.p')
poisoned_prompts = [pickle.load(open(f, 'rb')) for f in candidate_files]

# Paths
lora_weights_path = f'./output_models/{method}_300/lora_adapter'
poisoned_img_dir = "./poisoned_images/original/images"
clean_img_dir = "./poisoning_candidates/images"
baseline_model_path = "runwayml/stable-diffusion-v1-5"
poisoned_model_path = "runwayml/stable-diffusion-v1-5"

# Load Stable Diffusion pipelines
baseline_pipe = StableDiffusionPipeline.from_pretrained(baseline_model_path, torch_dtype=torch.float16, safety_checker=None).to(device)
baseline_pipe.set_progress_bar_config(disable=True)

poisoned_pipe = StableDiffusionPipeline.from_pretrained(
    poisoned_model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
)
poisoned_pipe.set_progress_bar_config(disable=True)
poisoned_pipe.to(device)

# Attach LoRA weights to the poisoned model
unet = poisoned_pipe.unet
unet = get_peft_model(unet, LoraConfig(
    r=4,
    target_modules=["to_k", "to_q", "to_v", "proj_out"],
    lora_alpha=16,
    lora_dropout=0.1,
))
lora_state_dict = torch.load(os.path.join(lora_weights_path, "pytorch_lora_weights.bin"), map_location="cpu")
set_peft_model_state_dict(unet, lora_state_dict)

def encode_image_tensor(img_tensor):
    with torch.no_grad():
        return clip_model.encode_image(img_tensor.to(device)).float().cpu().numpy()

def encode_image_path(img_path):
    try:
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
        return encode_image_tensor(img)
    except Exception as e:
        print(f"Failed to process image {img_path}: {e}")
        return None

def evaluate_images(poisoned_img_dir, clean_img_dir):
    scores = []
    for filename in os.listdir(poisoned_img_dir):
        img_id = filename.split('.')[0].split('_')[1]
        poisoned_path = os.path.join(poisoned_img_dir, filename)
        clean_path = os.path.join(clean_img_dir, f'dog_{img_id}.jpg')

        if not os.path.exists(clean_path):
            print(f"[SKIP] Clean image not found: {filename}")
            continue

        poison_vec = encode_image_path(poisoned_path)
        clean_vec = encode_image_path(clean_path)

        if poison_vec is None or clean_vec is None:
            print(f"[SKIP] Failed to encode: {filename}")
            continue

        sim = cosine_similarity(poison_vec, clean_vec)[0][0]
        scores.append(sim)

    if not scores:
        print("⚠️ No valid image comparisons made! Check if filenames match and images are valid.")

    return np.mean(scores), np.std(scores)

def evaluate_model(pipe, prompts):
    clip_scores = []
    for prompt_obj in tqdm(prompts, desc="Generating & scoring prompts"):
        prompt = prompt_obj["text"]
        try:
            gen_img = pipe(prompt, num_inference_steps=30).images[0]
        except Exception as e:
            print(f"Failed to generate image for prompt '{prompt}': {e}")
            continue

        # Get CLIP embeddings
        try:
            gen_tensor = preprocess(gen_img).unsqueeze(0)
            ref_img = Image.fromarray(prompt_obj["img"]).convert("RGB")
            ref_tensor = preprocess(ref_img).unsqueeze(0)
        except Exception as e:
            print(f"Error processing image for CLIP: {e}")
            continue

        gen_vec = encode_image_tensor(gen_tensor)
        ref_vec = encode_image_tensor(ref_tensor)

        sim = cosine_similarity(gen_vec, ref_vec)[0][0]
        clip_scores.append(sim)

    return np.mean(clip_scores), np.std(clip_scores)

def compute_image_metrics(poisoned_img_dir, clean_img_dir):
    lpips_scores = []
    ssim_scores = []
    psnr_scores = []
    l2_scores = []

    for filename in os.listdir(poisoned_img_dir):
        img_id = filename.split('.')[0].split('_')[1]
        poisoned_path = os.path.join(poisoned_img_dir, filename)
        clean_path = os.path.join(clean_img_dir, f'dog_{img_id}.jpg')

        if not os.path.exists(clean_path):
            continue

        try:
            clean = Image.open(clean_path).convert("RGB").resize((224, 224))
            poison = Image.open(poisoned_path).convert("RGB").resize((224, 224))
        except Exception:
            continue

        clean_np = np.asarray(clean).astype(np.float32) / 255.0
        poison_np = np.asarray(poison).astype(np.float32) / 255.0

        # SSIM & PSNR
        ssim_score = ssim(clean_np, poison_np, channel_axis=-1, data_range=1.0)
        psnr_score = psnr(clean_np, poison_np, data_range=1.0)

        # LPIPS
        clean_tensor = transforms.ToTensor()(clean).unsqueeze(0).to(device)
        poison_tensor = transforms.ToTensor()(poison).unsqueeze(0).to(device)
        lpips_score = lpips_model(clean_tensor, poison_tensor).item()

        # L2 noise norm
        l2 = np.linalg.norm(clean_np - poison_np)

        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
        lpips_scores.append(lpips_score)
        l2_scores.append(l2)

    return {
        "SSIM": (np.mean(ssim_scores), np.std(ssim_scores)),
        "PSNR": (np.mean(psnr_scores), np.std(psnr_scores)),
        "LPIPS": (np.mean(lpips_scores), np.std(lpips_scores)),
        "L2": (np.mean(l2_scores), np.std(l2_scores))
    }

# ---- Run evaluations ----
print("🔍 Evaluating poisoned images...")
img_sim_mean, img_sim_std = evaluate_images(poisoned_img_dir, clean_img_dir)
print(f"Average similarity (poisoned vs clean): {img_sim_mean:.4f} ± {img_sim_std:.4f}")

print("🎨 Evaluating poisoned model...")
poisoned_score_mean, poisoned_score_std = evaluate_model(poisoned_pipe, poisoned_prompts)
print(f"Poisoned model CLIP alignment: {poisoned_score_mean:.4f} ± {poisoned_score_std:.4f}")

print("🧪 Evaluating baseline model...")
baseline_score_mean, baseline_score_std = evaluate_model(baseline_pipe, poisoned_prompts)
print(f"Baseline model CLIP alignment: {baseline_score_mean:.4f} ± {baseline_score_std:.4f}")

print("\n📉 Poisoning effectiveness:")
print(f"↓ Difference in CLIP alignment: {(baseline_score_mean - poisoned_score_mean):.4f}")

print("📊 Computing visual similarity metrics...")
visual_metrics = compute_image_metrics(poisoned_img_dir, clean_img_dir)
for metric, (mean, std) in visual_metrics.items():
    print(f"{metric}: {mean:.4f} ± {std:.4f}")