import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
import clip
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from ..py_files.perturbation_methods import fgsm_penalty, pgd_penalty, nightshade_penalty
from ..py_files.nightshade import Nightshade
from diffusers import StableDiffusionPipeline
import sys

force_fgsm = False
force_pgd = False
force_nightshade = False

if len(sys.argv) < 4:
    print("Usage: python hyperparam_search.py <input_dir> <output_dir> <fgsm/pgd/nightshade>")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(input_dir):
    print(f"Input directory {input_dir} does not exist.")
    sys.exit(1)
os.makedirs(output_dir, exist_ok=True)

if 'fgsm' in sys.argv:
    force_fgsm = True
if 'pgd' in sys.argv:
    force_pgd = True
if 'nightshade' in sys.argv:
    force_nightshade = True


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

def load_subset(path, max_samples=10):
    files = [f for f in sorted(os.listdir(path)) if f.endswith(".p")][:max_samples]
    subset = []
    for f in files:
        with open(os.path.join(path, f), 'rb') as pf:
            data = pickle.load(pf)
            subset.append(data)
    return subset

images = load_subset(f'{input_dir}/pickle', max_samples=20)
print(f"Loaded {len(images)} images from the dataset.")

encoded_target_text = CLIP_MODEL.encode_text(clip.tokenize(["cat"]).to(DEVICE))
encoded_target_text /= encoded_target_text.norm(dim=-1, keepdim=True)
encoded_no_text = CLIP_MODEL.encode_text(clip.tokenize([""]).to(DEVICE))
encoded_no_text /= encoded_no_text.norm(dim=-1, keepdim=True)

def compute_clip_similarity(image, text_features):
    image_tensor = CLIP_PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = CLIP_MODEL.encode_image(image_tensor)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return (image_features @ text_features.T).item()

def evaluate_attack(original_img, poisoned_img):
    # Resize images to 224x224
    original_img_resized = original_img.resize((224, 224))
    poisoned_img_resized = poisoned_img.resize((224, 224))
    # Convert to arrays
    orig_arr = np.array(original_img_resized, dtype=np.float32) / 255.0
    poison_arr = np.array(poisoned_img_resized, dtype=np.float32) / 255.0

    # Metrics
    clip_score_target = compute_clip_similarity(poisoned_img, encoded_target_text)
    clip_score_original = compute_clip_similarity(poisoned_img, encoded_no_text)  # empty prompt for base concept

    l2_dist = np.linalg.norm(poison_arr - orig_arr)
    ssim_val = ssim(orig_arr, poison_arr, channel_axis=-1, data_range=1.0)
    psnr_val = psnr(orig_arr, poison_arr, data_range=1.0)

    return {
        "clip_target": clip_score_target,
        "clip_original": clip_score_original,
        "l2": l2_dist,
        "ssim": ssim_val,
        "psnr": psnr_val
    }

fgsm_eps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

pgd_eps = [0.05, 0.1, 0.2, 0.3]
pgd_step_size = [0.03, 0.05, 0.1]  
pgd_iterations = [10, 20, 50]

nightshade_eps = [0.1, 0.2, 0.3]
nightshade_iterations = [20, 50, 100, 150]

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
).to(DEVICE)
pipe._progress_bar_config={"disable": True}
target_text = 'cat'

# FGSM
ns = Nightshade('cat', DEVICE, None, pipe)
if not os.path.exists(f"{output_dir}/fgsm_results.txt") or os.path.getsize(f"{output_dir}/fgsm_results.txt") == 0 or force_fgsm:
    with open(f"{output_dir}/fgsm_results.txt", "w") as f:
        results = []
        for i, image_data in tqdm(enumerate(images), desc="Images"):
            original_img = Image.fromarray(image_data['img'])
            ns.penalty_method = fgsm_penalty
            progress_bar = tqdm(fgsm_eps, total=len(fgsm_eps), desc="FGSM Epsilon", leave=False)
            for eps in progress_bar:
                progress_bar.set_postfix_str(f"Current: {eps}")
                ns.eps = eps
                poisoned_img = ns.generate(original_img, target_text, f"{input_dir}/anchor_images/anchor_{i:04d}.jpg")
                metrics = evaluate_attack(original_img, poisoned_img)
                results.append((eps, metrics))
            
        # Average the results for each epsilon
        avg_results = {}
        for eps, metrics in results:
            if eps not in avg_results:
                avg_results[eps] = {k: [] for k in metrics.keys()}
            for k, v in metrics.items():
                avg_results[eps][k].append(v)
        for eps, metrics in avg_results.items():
            avg_results[eps] = {k: np.mean(v) for k, v in metrics.items()}
            f.write(f"FGSM Epsilon: {eps}\n")
            for k, v in avg_results[eps].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

# PGD
ns = Nightshade('cat', DEVICE, None, pipe)
if not os.path.exists(f"{output_dir}/pgd_results.txt") or os.path.getsize(f"{output_dir}/pgd_results.txt") == 0 or force_pgd:
    with open(f"{output_dir}/pgd_results.txt", 'w') as f:
        results = []
        for i, image_data in tqdm(enumerate(images)):
            original_img = Image.fromarray(image_data['img'])
            ns.penalty_method = pgd_penalty
            eps_progress_bar = tqdm(pgd_eps, total=len(pgd_eps), desc="PGD Epsilon", leave=False)
            
            for eps in eps_progress_bar:
                eps_progress_bar.set_postfix_str(f"Current: {eps}")
                step_size_progress_bar = tqdm(pgd_step_size, total=len(pgd_step_size), desc="PGD Step Size", leave=False)
                for step_size in step_size_progress_bar:
                    step_size_progress_bar.set_postfix_str(f"Current: {step_size}")
                    iterations_progress_bar = tqdm(pgd_iterations, total=len(pgd_iterations), desc="PGD Iterations", leave=False)
                    for iterations in iterations_progress_bar:
                        iterations_progress_bar.set_postfix_str(f"Current: {iterations}")
                        ns.eps = eps
                        ns.step_size = step_size
                        ns.iterations = iterations
                        poisoned_img = ns.generate(original_img, target_text, f"{input_dir}/anchor_images/anchor_{i:04d}.jpg")
                        metrics = evaluate_attack(original_img, poisoned_img)
                        results.append((eps, step_size, iterations, metrics))
            
        # Average the results for each combination of parameters
        avg_results = {}
        for eps, step_size, iterations, metrics in results:
            key = (eps, step_size, iterations)
            if key not in avg_results:
                avg_results[key] = {k: [] for k in metrics.keys()}
            for k, v in metrics.items():
                avg_results[key][k].append(v)
        for key, metrics in avg_results.items():
            avg_results[key] = {k: np.mean(v) for k, v in metrics.items()}
            f.write(f"PGD Epsilon: {key[0]}, Step Size: {key[1]}, Iterations: {key[2]}\n")
            for k, v in avg_results[key].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

# Original Nightshade penalty
ns = Nightshade('cat', DEVICE, None, pipe)
ns.verbose = False

if not os.path.exists(f"{output_dir}/nightshade_results.txt") or os.path.getsize(f"{output_dir}/nightshade_results.txt") == 0 or force_nightshade:
    with open(f"{output_dir}/nightshade_results.txt", 'w') as f:
        results = []
        for i, image_data in tqdm(enumerate(images)):
            original_img = Image.fromarray(image_data['img'])
            ns.penalty_method = nightshade_penalty
            eps_progress_bar = tqdm(nightshade_eps, total=len(nightshade_eps), desc="Nightshade Epsilon", leave=False)
            for eps in eps_progress_bar:
                eps_progress_bar.set_postfix_str(f"Current: {eps}")
                iterations_progress_bar = tqdm(nightshade_iterations, total=len(nightshade_iterations), desc="Nightshade Iterations", leave=False)
                for iterations in iterations_progress_bar:
                    iterations_progress_bar.set_postfix_str(f"Current: {iterations}")
                    ns.eps = eps
                    ns.t_size = iterations
                    poisoned_img = ns.generate(original_img, target_text, f"{input_dir}/anchor_images/anchor_{i:04d}.jpg")
                    metrics = evaluate_attack(original_img, poisoned_img)
                    results.append((eps, iterations, metrics))
            
        # Average the results for each combination of parameters
        avg_results = {}
        for eps, iterations, metrics in results:
            key = (eps, iterations)
            if key not in avg_results:
                avg_results[key] = {k: [] for k in metrics.keys()}
            for k, v in metrics.items():
                avg_results[key][k].append(v)
        for key, metrics in avg_results.items():
            avg_results[key] = {k: np.mean(v) for k, v in metrics.items()}
            f.write(f"Nightshade Epsilon: {key[0]}, Iterations: {key[1]}\n")
            for k, v in avg_results[key].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

