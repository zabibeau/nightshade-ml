import os
import torch
import clip
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import ImageReward as RM

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
lpips_model = lpips.LPIPS(net='alex').to(device)
image_reward_scorer = RM.load('ImageReward-v1.0')


# Load prompts from pickle files
candidate_files = glob.glob('./poisoning_candidates/pickle/*.p')
poisoned_prompts = [pickle.load(open(f, 'rb')) for f in candidate_files]


def encode_image_path(img_path):
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy(), image

def encode_texts(texts):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()


def compute_image_quality_metrics(poisoned_img_dir, clean_img_dir):
    lpips_scores = []
    ssim_scores = []
    psnr_scores = []
    l2_scores = []
    cosine_sims = []

    for filename in tqdm(os.listdir(poisoned_img_dir), desc="Computing image quality metrics", total=len(os.listdir(poisoned_img_dir))):
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

        clean_vec, _ = encode_image_path(clean_path)
        poison_vec, _ = encode_image_path(poisoned_path)
        similarity = cosine_similarity(clean_vec, poison_vec)[0][0]

        clean_np = np.asarray(clean).astype(np.float32) / 255.0
        poison_np = np.asarray(poison).astype(np.float32) / 255.0

        ssim_score = ssim(clean_np, poison_np, channel_axis=-1, data_range=1.0)
        psnr_score = psnr(clean_np, poison_np, data_range=1.0)

        clean_tensor = transforms.ToTensor()(clean).unsqueeze(0).to(device)
        poison_tensor = transforms.ToTensor()(poison).unsqueeze(0).to(device)
        lpips_score = lpips_model(clean_tensor, poison_tensor).item()

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
    poisoned_dog_sims = []
    poisoned_cat_sims = []
    clean_dog_sims = []
    clean_cat_sims = []

    for filename in tqdm(os.listdir(poisoned_img_dir), desc="Calculating semantic similarity", total=len(os.listdir(poisoned_img_dir))):
        img_id = filename.split('.')[0].split('_')[1]
        poisoned_path = os.path.join(poisoned_img_dir, filename)
        clean_path = os.path.join(clean_img_dir, f'dog_{img_id}.jpg')

        clean_clip, _ = encode_image_path(clean_path)
        poisoned_clip, _ = encode_image_path(poisoned_path)

        poisoned_dog = cosine_similarity(poisoned_clip.reshape(1, -1), dog_text.reshape(1, -1))[0][0]
        poisoned_cat = cosine_similarity(poisoned_clip.reshape(1, -1), cat_text.reshape(1, -1))[0][0]
        clean_dog = cosine_similarity(clean_clip.reshape(1, -1), dog_text.reshape(1, -1))[0][0]
        clean_cat = cosine_similarity(clean_clip.reshape(1, -1), cat_text.reshape(1, -1))[0][0]

        poisoned_dog_sims.append(poisoned_dog)
        poisoned_cat_sims.append(poisoned_cat)
        clean_dog_sims.append(clean_dog)
        clean_cat_sims.append(clean_cat)

    return {
        'Poisoned Dog Similarity': (np.mean(poisoned_dog_sims), np.std(poisoned_dog_sims)),
        'Poisoned Cat Similarity': (np.mean(poisoned_cat_sims), np.std(poisoned_cat_sims)),
        'Clean Dog Similarity': (np.mean(clean_dog_sims), np.std(clean_dog_sims)),
        'Clean Cat Similarity': (np.mean(clean_cat_sims), np.std(clean_cat_sims))
    }


import numpy as np

def compute_attack_effectiveness(gen_img_dir, clean_gen_img_dir):
    dog_text, cat_text = encode_texts(['a photo of a dog', 'a photo of a cat'])
    prompt = 'a photo of a dog'

    accumulators = {
        'clipscore': 0.0,
        'poisoned_image_reward': 0.0,
        'clean_image_reward': 0.0,
        'poisoned_dog': 0.0,
        'poisoned_cat': 0.0,
        'clean_dog': 0.0,
        'clean_cat': 0.0,
        'perceptual_drift': 0.0
    }

    count = 0
    for filename in tqdm(os.listdir(gen_img_dir), desc='Calculating attack effectiveness'):
        img_id = filename.split('.')[0].split('_')[1]
        poisoned_path = os.path.join(gen_img_dir, filename)
        clean_path = os.path.join(clean_gen_img_dir, f"none_{img_id}.png")

        poisoned_feat, poisoned_tensor = encode_image_path(poisoned_path)
        clean_feat, clean_tensor = encode_image_path(clean_path)

        poisoned_dog_sim = cosine_similarity(poisoned_feat.reshape(1, -1), dog_text.reshape(1, -1))[0][0]
        poisoned_cat_sim = cosine_similarity(poisoned_feat.reshape(1, -1), cat_text.reshape(1, -1))[0][0]
        clean_dog_sim = cosine_similarity(clean_feat.reshape(1, -1), dog_text.reshape(1, -1))[0][0]
        clean_cat_sim = cosine_similarity(clean_feat.reshape(1, -1), cat_text.reshape(1, -1))[0][0]

        with torch.no_grad():
            perceptual_drift = float(lpips_model(poisoned_tensor, clean_tensor))

        clipscore = poisoned_dog_sim

        image_reward_score_clean = image_reward_scorer.score(prompt, Image.open(clean_path).convert("RGB"))
        image_reward_score_poisoned = image_reward_scorer.score(prompt, Image.open(poisoned_path).convert("RGB"))

        # Accumulate sums
        accumulators['clipscore'] += clipscore
        accumulators['poisoned_image_reward'] += image_reward_score_poisoned
        accumulators['clean_image_reward'] += image_reward_score_clean
        accumulators['poisoned_dog'] += poisoned_dog_sim
        accumulators['poisoned_cat'] += poisoned_cat_sim
        accumulators['clean_dog'] += clean_dog_sim
        accumulators['clean_cat'] += clean_cat_sim
        accumulators['perceptual_drift'] += perceptual_drift

        count += 1

    # Compute averages
    averages = {k: v / count for k, v in accumulators.items()}
    return averages


def compute_gen_semantic_metrics(poisoned_img_dir, clean_img_dir):
    dog_text, cat_text = encode_texts(['a photo of a dog', 'a photo of a cat']) 
    poisoned_dog_sims = []
    poisoned_cat_sims = []
    clean_dog_sims = []
    clean_cat_sims = []

    for filename in tqdm(os.listdir(poisoned_img_dir), desc="Calculating semantic similarity", total=len(os.listdir(poisoned_img_dir))):
        img_id = filename.split('.')[0].split('_')[1]
        poisoned_path = os.path.join(poisoned_img_dir, filename)
        clean_path = os.path.join(clean_img_dir, f'none_{img_id}.png')

        clean_clip, _ = encode_image_path(clean_path)
        poisoned_clip, _ = encode_image_path(poisoned_path)

        poisoned_dog = cosine_similarity(poisoned_clip.reshape(1, -1), dog_text.reshape(1, -1))[0][0]
        poisoned_cat = cosine_similarity(poisoned_clip.reshape(1, -1), cat_text.reshape(1, -1))[0][0]
        clean_dog = cosine_similarity(clean_clip.reshape(1, -1), dog_text.reshape(1, -1))[0][0]
        clean_cat = cosine_similarity(clean_clip.reshape(1, -1), cat_text.reshape(1, -1))[0][0]

        poisoned_dog_sims.append(poisoned_dog)
        poisoned_cat_sims.append(poisoned_cat)
        clean_dog_sims.append(clean_dog)
        clean_cat_sims.append(clean_cat)

    return {
        'Poisoned Dog Similarity': (np.mean(poisoned_dog_sims), np.std(poisoned_dog_sims)),
        'Poisoned Cat Similarity': (np.mean(poisoned_cat_sims), np.std(poisoned_cat_sims)),
        'Clean Dog Similarity': (np.mean(clean_dog_sims), np.std(clean_dog_sims)),
        'Clean Cat Similarity': (np.mean(clean_cat_sims), np.std(clean_cat_sims))
    }


        