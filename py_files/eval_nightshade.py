import torch
import clip
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms

class NightshadeEvaluator:

    def __init__(self, device='cuda', img_size=512):
        self.img_size = img_size
        self.device = device
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpip = LearnedPerceptualImagePatchSimilarity().to(device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def evaluate_perturbation(self, original, perturbed):

        orig_tensor = self._img_to_tensor(original).to(self.device)
        pert_tensor = self._img_to_tensor(perturbed).to(self.device)

        metrics = {
            'l2_norm': (orig_tensor - pert_tensor).norm(p=2).item(),
            'linf_norm': (orig_tensor - pert_tensor).abs().max().item(),
            'ssim': self.ssim(orig_tensor, pert_tensor).item(),
            'lpips': self.lpip(orig_tensor, pert_tensor).item()
        }

        return metrics
    
    def evaluate_embedding_shift(self, original, perturbed, target_concept):
        
        with torch.no_grad():
            text_input = clip.tokenize([f'a photo of a {target_concept}']).to(self.device)
            text_features = self.clip_model.encode_text(text_input)

            orig_embed = self.clip_model.encode_image(self.clip_preprocess(original).unsqueeze(0).to(self.device))
            pert_embed = self.clip_model.encode_image(self.clip_preprocess(perturbed).unsqueeze(0).to(self.device))

            metrics = {
                'original_target_sim': torch.cosine_similarity(orig_embed, text_features).item(),
                'perturbed_target_sim': torch.cosine_similarity(pert_embed, text_features).item(),
                'embedding_shift': torch.norm(orig_embed - pert_embed, 2).item()
            }

            return metrics
        
    def evaluate_attack_effectiveness(self, clean_model, poisoned_model, test_prompts):
        results = {}
        for prompt in test_prompts:
            clean_output = clean_model(prompt).images[0]
            poisoned_output = poisoned_model(prompt).images[0]

            results[prompt] = {
                'visual_diff': self.evaluate_perturbation(clean_output, poisoned_output),
                'clip_sim': self.evaluate_embedding_shift(clean_output, poisoned_output, prompt)
            }
        return results
    
    def _img_to_tensor(self, img):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(img).unsqueeze(0).to(self.device)