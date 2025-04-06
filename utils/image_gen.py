
# utils/image_gen.py

from diffusers import StableDiffusionPipeline
import torch

# Load the model (requires internet the first time)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2", 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    # Save image temporarily
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path
