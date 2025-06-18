import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import os
from PIL import Image

def generate(prompt):
    # model_id = "CompVis/stable-diffusion-v1-4"
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to(device)

    size = 512

    image = pipe(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=size,
        width=size,
    ).images[0]

    image.save("./output.png")
    return image



def transform(init_image, prompt):
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    size = 512

    init_image = init_image.resize((size, size), Image.LANCZOS)


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
    return image