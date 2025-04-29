import json
import clip
import pandas as pd
import torch
import os
import glob
import pickle
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from tqdm.notebook import tqdm
from collections import defaultdict


# Note: this code is heavily based on the original nightshade implementation which can be found at:
# https://github.com/Shawn-Shan/nightshade-release/tree/main

def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)

class ClipModel(object):

    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.cuda()
        self.tokenizer = clip.tokenize

    def get_text_embedding(self, text):
        if isinstance(text, str):
            text = [text]
        tokenized_text = self.tokenizer(text, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_text)
        return text_features
    
    def get_image_embedding(self, img):
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features
    
    def get_score(self, images, texts, softmax=False):
        """Process batch of images and texts"""
        if isinstance(texts, str):
            texts = [texts] * len(images)
            
        # Batch image processing
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_inputs = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            text_features = self.model.encode_text(text_inputs)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (text_features @ image_features.T).cpu().numpy()
            return np.diag(similarity)  # Return pairwise scores

            
def get_dataset(annotation_file, data_dir, limit=None, unique_images=True):
    """
    Args:
        unique_images: If True, returns only one caption per image (randomly selected)
    """
    annotations = json.load(open(annotation_file, 'r'))
    
    # Group by image_id if we want unique images
    if unique_images:
        image_dict = defaultdict(list)
        for ann in annotations['annotations']:
            image_dict[ann['image_id']].append(ann)
        
        # Randomly select one caption per image
        data = []
        for img_id, anns in tqdm(image_dict.items(), desc="Processing unique images", total=len(image_dict)):
            selected = random.choice(anns)
            img_path= f"{data_dir}/COCO_train2014_{img_id:012d}.jpg"
            data.append({
                'annotation_id': selected['id'],
                'image_id': img_id,
                'caption': selected['caption'],
                'image_path': img_path
            })

    else:
        data = []
        for ann in tqdm(annotations['annotations'], desc="Processing all captions", total=len(annotations['annotations'])):
            img_path = f"{data_dir}/COCO_train2014_{ann['image_id']:012d}.jpg"
            data.append({
                'annotation_id': ann['id'],
                'image_id': ann['image_id'],
                'caption': ann['caption'],
                'image_path': img_path,
            })
    
    df = pd.DataFrame(data)
    if limit:
        df = df.sample(n=limit, random_state=5806)
    
    print(f"Loaded {len(df)} {'unique image' if unique_images else 'caption'} entries")
    return df

def get_poisoning_candidates(df, concept, num_candidates=300, clip_threshold=0.25, output_dir='poisoning_candidates', batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = ClipModel(device)
    os.makedirs(output_dir, exist_ok=True)
    
    candidates = []  # Will store (image, caption, image_id) tuples
    text_prompt = f"a photo of a {concept}"

    # Process images in batches
    for batch_start in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        batch_images = []
        batch_entries = []
        
        # Load and preprocess batch
        for _, row in batch_df.iterrows():
            try:
                img = Image.open(row['image_path']).convert("RGB")
                img = crop_to_square(img)
                batch_images.append(img)
                batch_entries.append(row)
            except Exception as e:
                print(f"Error loading image {row['image_path']}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # Batch scoring
        scores = clip_model.get_score(batch_images, [text_prompt]*len(batch_images))
        
        # Filter candidates
        for score, row, img in zip(scores, batch_entries, batch_images):
            if score > clip_threshold:
                candidates.append((img, row['caption'], row['image_id']))
    
    # Check if we have enough candidates
    if len(candidates) < num_candidates:
        raise Exception(f"Not enough candidates for {concept}, found {len(candidates)}, required {num_candidates}")
    
    # Extract components for embedding calculation
    candidate_images, candidate_captions, candidate_ids = zip(*candidates)
    
    # Batch compute text embeddings
    with torch.no_grad():
        # Process all captions at once
        caption_embeddings = clip_model.get_text_embedding(candidate_captions).cpu().numpy()
        target_embedding = clip_model.get_text_embedding([text_prompt]).cpu().numpy()
    
    # Calculate similarities in bulk
    sims = cosine_similarity(caption_embeddings, target_embedding).flatten()
    top_indices = np.argsort(sims)[::-1][:num_candidates]
    
    # Save top candidates
    for i, idx in enumerate(tqdm(top_indices, desc="Saving candidates")):
        img, caption, image_id = candidates[idx]
        current_data = {
            "img": np.array(img),
            "text": caption,
            "image_id": image_id
        }
        pickle.dump(current_data, open(os.path.join(output_dir, f"{concept}_{i}.p"), 'wb'))
    
    print(f"Saved {len(top_indices)} poisoning candidates for {concept} in {output_dir}")

def create_mixed_dataset(clean_df, poisoned_df):
    if len(poisoned_df) > len(clean_df):
        raise ValueError("Poisoned dataset cannot be larger than clean dataset.")
    if 'image_id' not in clean_df.columns or 'image_id' not in poisoned_df.columns:
        raise ValueError("Both datasets must contain 'image_id' column.")
    if len(clean_df) == 0 or len(poisoned_df) == 0:
        raise ValueError("Both datasets must be non-empty.")

    clean_df = clean_df.copy()
    poisoned_df = poisoned_df.copy()

    used_indices = set()  # Keep track of already-replaced indices

    for i, row in tqdm(poisoned_df.iterrows(), desc="Mixing datasets", total=len(poisoned_df)):
        # Find matching row in clean_df
        matching_indices = clean_df.index[clean_df['image_id'] == row['image_id']]

        if len(matching_indices) > 0:
            # If ID exists, replace that entry
            idx = matching_indices[0]
        else:
            # Otherwise, pick a random clean sample not already used
            available_indices = list(set(clean_df.index) - used_indices)
            if not available_indices:
                raise RuntimeError("No available clean samples left to replace!")
            idx = random.choice(available_indices)

        used_indices.add(idx)

        # Update each column individually
        for col in ['caption', 'image_path', 'image_id']:
            clean_df.at[idx, col] = row[col]

    mixed_df = clean_df.sample(frac=1).reset_index(drop=True)  # Shuffle
    print(f"✅ Created mixed dataset with {len(poisoned_df)}/{len(mixed_df)} poisoned entries ({len(poisoned_df) / len(mixed_df) * 100:.2f}% poisoned).")

    return mixed_df


def get_poisoned_dataset(poisoning_candidates_dir, limit=None):
    poisoned_df = []
    for file in glob.glob(os.path.join(poisoning_candidates_dir, "*.p")):
        data = pickle.load(open(file, 'rb'))
        poisoned_df.append({
            'image_id': file.split('/')[-1].split('_')[1].split('.')[0],
            'caption': data['text'],
            'image_path': file,
        })
    poisoned_df = pd.DataFrame(poisoned_df)
    if limit:
        poisoned_df = poisoned_df.sample(n=limit, random_state=5806)
    print(f"Loaded {len(poisoned_df)} poisoned entries from {poisoning_candidates_dir}")
    return poisoned_df