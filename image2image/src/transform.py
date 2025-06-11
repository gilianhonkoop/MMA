import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import make_image_grid, load_image
import os
from PIL import Image

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()


image_path = os.path.join(os.path.dirname(__file__), "images", "mars.png")
size = 512
init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((size, size), Image.LANCZOS)

prompt = ("Transform the sand to look like it's covered in natural moss or grass — green and lush. "
          "Keep the astronaut, horse, sky, lighting, and everything else exactly the same.")

image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.2,
    # num_inference_steps=50,
    # guidance_scale=8,
    image_guidance_scale=0.25,
    # generator=generator,
).images[0]

out = os.path.join(os.path.dirname(__file__), "output", "edited_image.png")
image.save(out)
