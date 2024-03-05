from beam import App, Runtime, Image, Output, Volume

import os
import torch
import PIL
from diffusers import AutoPipelineForInpainting, AutoPipelineForImage2Image, ControlNetModel
from diffusers.utils import load_image
import base64
from io import BytesIO

cache_path = "./models"

# The environment your app runs on
app = App(
    name="stable-diffusion-app",
    runtime=Runtime(
        cpu=1,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.8",
            python_packages=[
                "diffusers[torch]>=0.10",
                "transformers",
                "torch",
                "pillow",
                "accelerate",
                "safetensors",
                "xformers"
            ],
        ),
    ),
    volumes=[Volume(name="models", path="./models")],
)

def decode_base64_image(image_string):
    image_string = image_string[len("data:image/png;base64,"):]
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    image = PIL.Image.open(buffer)
    rgb = image.convert('RGB')
    return rgb

# Temp: load images for testing
img = load_image(
    "./example_image.png"
)

mask_image = load_image(
    "./example_mask.png"
)

control_image = load_image(
    "./example_control_image.png"
)

# This runs once when the container first boots
def load_models():
    torch.backends.cuda.matmul.allow_tf32 = True

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16
    )

    inPaintPipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir=cache_path,
    ).to("cuda")

    sdxlInPaintPipe = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir=cache_path,
    ).to("cuda")

    sdxlImg2ImgPipe = AutoPipelineForImage2Image.from_pipe(sdxlInPaintPipe)

    inPaintPipe.enable_xformers_memory_efficient_attention()
    sdxlInPaintPipe.enable_xformers_memory_efficient_attention()
    sdxlImg2ImgPipe.enable_xformers_memory_efficient_attention()

    return (inPaintPipe, sdxlInPaintPipe, sdxlImg2ImgPipe)

@app.task_queue(
    loader=load_models,
    outputs=[Output(path="output.png")],
)
def generate_image(**inputs):
    
    # prompt = inputs["prompt"]

    # mask_image = decode_base64_image(
    #     inputs["mask_image"]
    # )

    # control_image = decode_base64_image(
    #     inputs["control_image"]
    # )

    prompt = "A man holding a camera"
    
    # Retrieve pre-loaded models from loader
    (inPaintPipe, sdxlInPaintPipe, sdxlImg2ImgPipe) = inputs["context"]

    image = inPaintPipe(
        prompt=prompt,
        image=img,
        mask_image=mask_image,
        control_image=control_image,
        guidance_scale=8.0,
        num_inference_steps=20
    ).images[0]

    image = sdxlInPaintPipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=100,
        strength=0.2,
        output_type="latent",  # keep in latent to save some VRAM
    ).images[0]

    image = sdxlImg2ImgPipe(
        prompt=prompt,
        image=image,
        guidance_scale=8.0,
        num_inference_steps=100,
        strength=0.2,
    ).images[0]

    print(f"Saved Image: {image}")
    image.save("output.png")