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
    
    def get_score(self, image, text, softmax=False):
        if isinstance(text, str):
            text = [text]

        if isinstance(image, list):
            image = [self.preprocess(i).unsqueeze(0).to(self.device) for i in image]
            image = torch.concat(image)
        else:
            image = self.preprocess(image).unsqueeze(0).to(self.device)

        text = self.tokenizer(text).to(self.device)

        if softmax:
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            return probs
        else:
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
                s = similarity[0][0]

            return s

            
def get_dataset(annotation_file, data_dir, limit=None, unique_images=True):
    """
    Args:
        unique_images: If True, returns only one caption per image (randomly selected)
    """
    annotations = json.load(open(annotation_file, 'r'))
    
    # Group by image_id if we want unique images
    if unique_images:
        from collections import defaultdict
        image_dict = defaultdict(list)
        for ann in annotations['annotations']:
            image_dict[ann['image_id']].append(ann)
        
        # Randomly select one caption per image
        data = []
        for img_id, anns in image_dict.items():
            selected = random.choice(anns)
            data.append({
                'annotation_id': selected['id'],
                'image_id': img_id,
                'caption': selected['caption'],
                'image_path': f"{data_dir}/COCO_train2014_{img_id:012d}.jpg"
            })
    else:
        data = [{
            'annotation_id': ann['id'],
            'image_id': ann['image_id'],
            'caption': ann['caption'],
            'image_path': f"{data_dir}/COCO_train2014_{ann['image_id']:012d}.jpg"
        } for ann in annotations['annotations']]
    
    df = pd.DataFrame(data)
    if limit:
        df = df.head(limit)
    
    print(f"Loaded {len(df)} {'unique image' if unique_images else 'caption'} entries")
    return df

def get_poisoning_candidates(df, concept, num_candidates=300, clip_threshold=0.25, output_dir='poisoning_candidates'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model = ClipModel(device)
    os.makedirs(output_dir, exist_ok=True)
    candidates = []

    for _, row in tqdm(df.iterrows(), desc="Processing images", total=len(df)):
        try:
            img = Image.open(row['image_path']).convert("RGB")
            img = crop_to_square(img)
            score = clip_model.get_score(img, f"a photo of a {concept}")
            if score > clip_threshold:
                candidates.append((img, row['caption'], row['image_id']))
        except Exception as e:
            print(f"Error processing image {row['image_path']}: {e}")
            continue
    if len(candidates) < num_candidates:
        raise Exception(f"Not enough candidates for {concept}, found {len(candidates)}, required {num_candidates}")
    
    captions = [c[1] for c in candidates]
    caption_embeddings = clip_model.get_text_embedding(captions).cpu().numpy()
    target_embedding = clip_model.get_text_embedding(f"a photo of a {concept}").cpu().numpy()
    sims = cosine_similarity(caption_embeddings, target_embedding).flatten()
    top_candidates = np.argsort(sims)[::-1][:num_candidates]

    for i, idx in tqdm(enumerate(top_candidates), desc=f"Saving candidates for {concept}", total=len(top_candidates)):
        img, caption, image_id = candidates[idx]
        current_data = {
            "img": np.array(img),
            "text": caption,
            "image_id": image_id
        }
        pickle.dump(current_data, open(os.path.join(output_dir, f"{concept}_{i}.p"), 'wb'))

    print(f"Saved {len(top_candidates)} poisoning candidates for {concept} in {output_dir}")