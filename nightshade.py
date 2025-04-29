from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
from einops import rearrange
from torchvision import transforms

SD_MODEL = "runwayml/stable-diffusion-v1-5"


# Note: parts of this code are heavily based on the original nightshade implementation which can be found at:
# https://github.com/Shawn-Shan/nightshade-release/tree/main

class Nightshade:
    
    def __init__(self, target_concept, device,  penalty_method=None, sd_pipeline=None, eps=0.1):
        self.target_concept = target_concept
        self.device = device
        self.eps = eps

        if sd_pipeline is None:
            self.sd_pipeline = self.get_model()
        else:
            self.sd_pipeline = sd_pipeline
            
        self.transform = self._create_transforms()
        if penalty_method is None:
            raise ValueError("Please provide a valid penalty method.")
        else:
            self.penalty_method = penalty_method
        self.latent_function = torch.compile(self.get_latent, mode='reduce-overhead')


    def get_model(self):
        # Load the Stable Diffusion model
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL,
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        pipe.to(self.device)
        pipe._progress_bar_config = {"disable": True}
        return pipe

    def _create_transforms(self):
        # Create the image transforms to resize and crop images to 512x512
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
        return image_transforms
    
    def get_latent(self, tensor):
        # Convert the image tensor to latent space using the VAE of the Stable Diffusion model
        latent = self.sd_pipeline.vae.encode(tensor).latent_dist.mean
        return latent
    
    def get_penalty(self, source_tensor, target_latent, modifier):
        
        # Generate the perturbation using the penalty method
        perturbation = self.penalty_method(
            source_tensor, 
            target_latent, 
            modifier, 
            self.latent_function, 
            eps=self.eps
        )
        # Clamp the perturbation to ensure pixel values are within valid range
        return torch.clamp(perturbation + source_tensor, -1.0, 1.0)

    def convert_to_tensor(self, cur_img):
        # Convert the input image to a tensor
        cur_img = cur_img.resize((512, 512), resample=Image.BICUBIC)
        np_img = (np.array(cur_img).astype(np.float32) / 127.5) - 1.0
        tensor_img = torch.from_numpy(rearrange(np_img, 'h w c -> c h w')).unsqueeze(0).pin_memory()
        return tensor_img

    def convert_to_image(self, cur_img):
        # Conver the tensor back into an image
        if cur_img.dim() == 3:
            cur_img = cur_img.unsqueeze(0)

        cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
        np_img = (cur_img[0].cpu().numpy() * 255).astype(np.uint8)
        np_img = rearrange(np_img, 'c h w -> h w c')
        return Image.fromarray(np_img)
    
    def generate(self, img, target_concept):
        # Generate target image
        with torch.no_grad():
            target_image = self.generate_target(f"A photo of a {target_concept}")

        resized_img = self.transform(img)

        # Source and target tensor
        source_tensor = self.convert_to_tensor(resized_img).to(self.device, non_blocking=True).half()
        target_tensor = self.convert_to_tensor(target_image).to(self.device, non_blocking=True).half()

        # Encode target latent
        with torch.no_grad():
            target_latent = self.get_latent(target_tensor)

        modifier = torch.zeros_like(source_tensor)

        final_adv = self.get_penalty(source_tensor, target_latent, modifier)

        return self.convert_to_image(final_adv)

    def generate_target(self, prompt, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            target_imgs = self.sd_pipeline(
                prompt,
                guidance_scale=7.5,
                num_inference_steps=25,
                height=512,
                width=512
            ).images
        return target_imgs[0]