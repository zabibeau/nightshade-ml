import clip
import torch
import os
import glob
import pickle
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

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
        self.model.to(self.device)
        self.tokenizer = clip.tokenize

    def get_text_embedding(self, text):
        if isinstance(text, str):
            text = [text]
        text = self.tokenizer(text, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features
    
    def get_image_embedding(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features
    
    def get_score(self, image, text, softmax=False):
        if isinstance(text, str):
            text = [text]
        
        if isinstance(image, list):
            image = [self.preprocess(img).unsqueeze(0).to(self.device) for img in image]
            image = torch.cat(image, dim=0)
        else:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        text = self.tokenizer(text, truncate=True).to(self.device)
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
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
                return similarity[0][0]
            

def DataProcess():
    
    def __init__(self, device):
        self.clip_model = ClipModel(device)
        self.device = device

    def process_data(self, directory, concept, outdir, num):
        data_dir = directory
        concept = concept
        os.makedirs(outdir, exist_ok=True)
        all_data = glob.glob(os.path.join(data_dir, '*.p'))
        res = []

        # Load data from specified file
        for i, cur_file in enumerate(all_data):
            data = pickle.load(open(cur_file, 'rb'))
            img = Image.fromarray(data['img'])
            text = data['text']

            img = crop_to_square(img)
            score = self.clip_model.get_score(img, f"a photo of a {concept}")
            # Make sure that the prompt somewhat matches the photo
            if score > 0.25:
                res.append((img, text))

        if len(res) < num:
            raise Exception(f"Not enough data for {concept}, found {len(res)}, required {num}")
        
        # Get text embeddings for all prompts and targets
        all_prompts = [d[1] for d in res]
        text_embeddings = self.clip_model.get_text_embedding(all_prompts)
        text_embeddings_target = self.clip_model.get_text_embedding(f"a photo of a {concept}")
        text_embeddings = text_embeddings.cpu().float().numpy()
        text_embeddings_target = text_embeddings_target.cpu().float().numpy()

        # Calculate cosine similarity between prompt and target embeddings and select the top 300 candidates
        sims = cosine_similarity(text_embeddings, text_embeddings_target).reshape(-1)
        candidates = np.argsort(sims)[::-1][:300]
        random_selected_candidate = random.sample(list(candidates), num)
        final_list = [res[i] for i in random_selected_candidate]
        for i, data in enumerate(final_list):
            img, text = data
            current_data = {
                "img": np.array(img),
                "text": text
            }
            pickle.dump(current_data, open(os.path.join(outdir, f"{concept}_{i}.p"), 'wb'))