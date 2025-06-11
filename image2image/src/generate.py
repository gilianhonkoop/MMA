import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline

# model_id = "CompVis/stable-diffusion-v1-4"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on Mars"

size = 512

image = pipe(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=size,
    width=size,
).images[0]

image.save("./output.png")
