from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
from einops import rearrange
from torchvision import transforms

SD_MODEL = "stabilityai/stable-diffusion-2-1"

class Nightshade:
    
    def __init__(self, target_concept, device, eps=0.1, penalty_method=None):
        self.target_concept = target_concept
        self.device = device
        self.eps = eps
        self.sd_pipeline = self.get_model()
        self.transform = self._create_transforms()
        self.penalty_method = penalty_method



    def get_model(self):
        # Load the Stable Diffusion model
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL,
            revision="fp16",
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        pipe.to(self.device)
        return pipe

    def _create_transforms(self):
        return transforms.Compose([
            transforms.Resize(512, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def get_latent(self, tensor):
        latent = self.sd_pipeline.vae.encode(tensor).latent_dist.mean
        return latent
    
    def get_penalty(self, source_tensor, target_latent, modifier):
        perturbation = self.penalty_method(
            source_tensor, 
            target_latent, 
            modifier, 
            self.get_latent, 
            eps=self.eps
        )
        final_penalty_tensor = torch.clamp(perturbation + source_tensor, -1, 1)
        return final_penalty_tensor

    def convert_to_tensor(self, cur_img):
        cur_img = cur_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
        cur_img = np.array(cur_img)
        img = (cur_img / 127.5 - 1.0).astype(np.float32)
        img = rearrange(img, 'h w c -> c h w')
        img = torch.tensor(img).unsqueeze(0)
        return img


    def convert_to_image(self, cur_img):
        if len(cur_img) == 512:
            cur_img = cur_img.unsqueeze(0)

        cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
        cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
        cur_img = Image.fromarray(cur_img.astype(np.uint8))
        return cur_img
    
    def generate(self, img, target_concept):
        resized_img = self.preprocess_image(img)
        source_tensor = self.convert_to_tensor(resized_img).to(self.device)
        target_image = self.generate_target(f"A photo of a {target_concept}")
        target_tensor = self.convert_to_tensor(target_image).to(self.device)

        with torch.no_grad():
            target_latent = self.get_latent(target_tensor.half())

        modifier = torch.zeros_like(source_tensor.half())
        source_tensor = source_tensor.half()

        final_adv = self.get_penalty(source_tensor, target_latent, modifier)

        return self.convert_to_image(final_adv)

    def generate_target(self, prompts):
        torch.manual_seed(5806)
        with torch.no_grad():
            target_imgs = self.sd_pipeline(
                prompts,
                guidance_scale=7.5,
                num_inference_steps=50,
                height=512,
                width=512,
            ).images
            target_imgs[0].save("target_image.png")
            return target_imgs[0]
