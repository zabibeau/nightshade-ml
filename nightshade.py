from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
from einops import rearrange
from torchvision import transforms

SD_MODEL = "stabilityai/stable-diffusion-2-1"


# Note: parts of this code are heavily based on the original nightshade implementation which can be found at:
# https://github.com/Shawn-Shan/nightshade-release/tree/main

class Nightshade:
    
    def __init__(self, target_concept, device, eps=0.1, penalty_method=None):
        self.target_concept = target_concept
        self.device = device
        self.eps = eps
        self.sd_pipeline = self.get_model()
        self.transform = self._create_transforms()
        if penalty_method is None:
            raise ValueError("Please provide a valid penalty method.")
        else:
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

        source_tensor = source_tensor.to(self.device)
        target_latent = target_latent.to(self.device)
        modifier = modifier.to(self.device)
        
        # Generate the perturbation using the penalty method
        perturbation = self.penalty_method(
            source_tensor, 
            target_latent, 
            modifier, 
            self.get_latent, 
            eps=self.eps
        )
        # Clamp the perturbation to ensure pixel values are within valid range
        final_penalty_tensor = torch.clamp(perturbation + source_tensor, -1.0, 1.0)
        return final_penalty_tensor

    def convert_to_tensor(self, cur_img):
        # Convert the input image to a tensor
        cur_img = cur_img.resize((512, 512), resample=Image.BICUBIC)
        cur_img = np.array(cur_img)
        img = (cur_img / 127.5 - 1.0).astype(np.float32)
        img = rearrange(img, 'h w c -> c h w')
        img = torch.tensor(img, device=self.device).unsqueeze(0)
        return img

    def convert_to_image(self, cur_img):
        # Conver the tensor back into an image
        if len(cur_img) == 512:
            cur_img = cur_img.unsqueeze(0)

        cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
        cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
        cur_img = Image.fromarray(cur_img.astype(np.uint8))
        return cur_img
    
    def generate(self, img, target_concept):
        # Generate an adversarial image based on the input image and target concept

        target_image = self.generate_target(f"A photo of a {target_concept}")
        resized_img = self.transform(img)

        source_tensor = self.convert_to_tensor(resized_img).to(self.device)
        target_tensor = self.convert_to_tensor(target_image).to(self.device)

        source_tensor = source_tensor.half()
        target_tensor = target_tensor.half()

        with torch.no_grad():
            target_latent = self.get_latent(target_tensor)

        modifier = torch.clone(source_tensor) * 0.0


        # Get the modified image based on the source tensor, target latent, and modifier
        final_adv = self.get_penalty(source_tensor, target_latent, modifier)

        return self.convert_to_image(final_adv)

    def generate_target(self, prompt, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            target_imgs = self.sd_pipeline(prompt, guidance_scale=7.5, num_inference_steps=50,
                                            height=512, width=512).images
        return target_imgs[0]