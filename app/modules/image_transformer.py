import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import os
from PIL import Image

class ImageTransformer:
    def __init__(self, model = "timbrooks/instruct-pix2pix"):
        self.model = model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model, torch_dtype=torch.bfloat16, variant="fp16", safety_checker=None
        ).to(device)

        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

    def transform(self, init_image, prompt, guidance_scale=7.5, image_guidance_scale=1.5, custom_size=False, image_size=(1024, 1024)):
        """
        Transform an image based on a prompt using the InstructPix2Pix model.

        Args:
            init_image (PIL.Image): The initial image to transform.
            prompt (str): The text prompt for the transformation.
            guidance_scale (float): Strength of the transformation.
            image_guidance_scale (float): Scale for image guidance.
            save_path (str): Path to save the transformed image. If None, the image will not be saved.
            custom_size (bool): Whether to use a custom image size.
            image_size (tuple): The size to resize the image to if custom_size is True.

        Returns:
            PIL.Image: The transformed image.
        """
        if not isinstance(init_image, Image.Image):
            raise ValueError("init_image must be a PIL.Image object")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if not isinstance(guidance_scale, (int, float)) or guidance_scale <= 1:
            raise ValueError("guidance_scale must be a number larger or equal to 1")
        if not isinstance(image_guidance_scale, (int, float)) or image_guidance_scale <= 1:
            raise ValueError("image_guidance_scale must be a number larger or equal to 1")
        if not isinstance(custom_size, bool):
            raise ValueError("custom_size must be a boolean")
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("image_size must be a tuple of two integers (width, height)")

        # Preferred size for the image as per Stable Diffusion docs
        default_size = 1024
        # Default in the pipeline is 100, this is for (temporary) faster inference. 
        # A higher value increases quality.
        num_inference_steps=50

        if custom_size:
            init_image = init_image.resize(image_size, Image.LANCZOS)
        else:
            init_image = init_image.resize((default_size, default_size), Image.LANCZOS)

        image = self.pipe(
            prompt=prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
        ).images[0]

        return image
