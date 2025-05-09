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
import evaluate
import ImageReward

method = sys.argv[1] if len(sys.argv) > 1 else "fgsm_300"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
lpips_model = lpips.LPIPS(net='alex').to(device)
fid = evaluate.load('fid')
is_metrics = evaluate.load('inception_score')
image_reward_scorer = ImageReward("ImageReward/ImageReward-v1.0", device=device)


# Load prompts from pickle files
candidate_files = glob.glob('./poisoning_candidates/pickle/*.p')
poisoned_prompts = [pickle.load(open(f, 'rb')) for f in candidate_files]


def encode_image_path(img_path):
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features, image

def encode_texts(texts):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def evaluate_images(poisoned_img_dir, clean_img_dir):
    scores = []
    for filename in os.listdir(poisoned_img_dir):
        img_id = filename.split('.')[0].split('_')[1]
        poisoned_path = os.path.join(poisoned_img_dir, filename)
        clean_path = os.path.join(clean_img_dir, f'dog_{img_id}.jpg')

        if not os.path.exists(clean_path):
            print(f"[SKIP] Clean image not found: {filename}")
            continue

        poison_vec, _ = encode_image_path(poisoned_path)
        clean_vec, _ = encode_image_path(clean_path)

        if poison_vec is None or clean_vec is None:
            print(f"[SKIP] Failed to encode: {filename}")
            continue

        sim = cosine_similarity(poison_vec, clean_vec)[0][0]
        scores.append(sim)

    if not scores:
        print("No valid image comparisons made! Check if filenames match and images are valid.")

    return np.mean(scores), np.std(scores)

def compute_image_quality_metrics(poisoned_img_dir, clean_img_dir):
    lpips_scores = []
    ssim_scores = []
    psnr_scores = []
    l2_scores = []
    cosine_sims = []

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

        # Cosine Similarity
        clean_vec, _ = encode_image_path(clean_path)
        poison_vec, _ = encode_image_path(poisoned_path)
        similarity = cosine_similarity(clean_vec, poison_vec)[0][0]

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

        cosine_sims.append(similarity)
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
        lpips_scores.append(lpips_score)
        l2_scores.append(l2)

    return {
        "Cosine Sim": (np.mean(cosine_sims), np.std(cosine_sims)),
        "SSIM": (np.mean(ssim_scores), np.std(ssim_scores)),
        "PSNR": (np.mean(psnr_scores), np.std(psnr_scores)),
        "LPIPS": (np.mean(lpips_scores), np.std(lpips_scores)),
        "L2": (np.mean(l2_scores), np.std(l2_scores))
    }

def compute_semantic_metrics(poisoned_img_dir, clean_img_dir):
    dog_text, cat_text = encode_texts(['a photo of a dog', 'a photo of a cat'])
    results = []
    for filename in tqdm(os.listdir(poisoned_img_dir), desc="Calculating semantic similarity"):
        img_id = filename.split('.')[0].split('_')[1]
        poisoned_path = os.path.join(poisoned_img_dir, filename)
        clean_path = os.path.join(clean_img_dir, f'dog_{img_id}.jpg')

        clean_clip = encode_image_path(clean_path)
        poisoned_clip = encode_image_path(poisoned_path)

        poisoned_dog = cosine_similarity(poisoned_clip, dog_text)
        poisoned_cat = cosine_similarity(poisoned_clip, cat_text)
        clean_dog = cosine_similarity(clean_clip, dog_text)
        clean_cat = cosine_similarity(clean_clip, cat_text)

        results.append({
            'clean_dog_score': clean_dog.item(),
            'clean_cat_score': clean_cat.item(),
            'poisoned_dog_score': poisoned_dog.item(),
            'poisoned_cat_score': poisoned_cat.item()
        })
    return results

def compute_attack_effectiveness(gen_img_dir, clean_gen_img_dir):
    results = []
    dog_text, cat_text = encode_texts(['a photo of a dog', 'a photo of a cat'])
    for filename in tqdm(os.listdir(gen_img_dir), desc='Calculating attack effectiveness'):
        img_id = filename.split('.')[0].split('_')[1]
        poisoned_path = os.path.join(gen_img_dir, filename)
        clean_path = os.path.join(clean_gen_img_dir, f"img_{img_id}.jpg")

        poisoned_feat, poisoned_tensor = encode_image_path(poisoned_path)
        clean_feat, clean_tensor = encode_image_path(clean_path)

        poisoned_dog_sim = cosine_similarity(poisoned_feat, dog_text)
        poisoned_cat_sim = cosine_similarity(poisoned_feat, cat_text)
        clean_dog_sim = cosine_similarity(clean_feat, dog_text)
        clean_cat_sim = cosine_similarity(clean_feat, cat_text)

        with torch.no_grad():
            perceptual_drift = float(lpips_model(poisoned_tensor, clean_tensor))

        clipscore = poisoned_dog_sim

        img_pil = transforms.ToPILImage()(poisoned_tensor.squeeze(0).cpu()).convert("RGB")
        image_reward_score = image_reward_scorer.score_pil(img_pil, 'a photo of a dog')

        results.append({
            'clipscore': clipscore,
            'image_reward': image_reward_score,
            'poisoned_dog': poisoned_dog_sim,
            'poisoned_cat': poisoned_cat_sim,
            'clean_dog': clean_dog_sim,
            'clean_cat': clean_cat_sim,
            'perceptual_drift': perceptual_drift
        })

    return results

        