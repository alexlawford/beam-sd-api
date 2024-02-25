from beam import App, Runtime, Image, Output, Volume

import os
import torch
import PIL
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
import base64
from io import BytesIO

cache_path = "./models"
model_id = "runwayml/stable-diffusion-v1-5"
controlnet_id = "lllyasviel/sd-controlnet-openpose"

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
mask_image = load_image(
    "./example_mask_image.png"
)

control_image = load_image(
    "./example_control_image.png"
)

# This runs once when the container first boots
def load_models():
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    return pipe

@app.task_queue(
    loader=load_models,
    outputs=[Output(path="output.png")],
)
def generate_image(**inputs):
    
    # Grab inputs passed to the API
    try:
        prompt = inputs["prompt"]
    # Use a default prompt if none is provided
    except KeyError:
        prompt = "A middle-aged man standing next to a polar bear in the artic"

    init_image = decode_base64_image(
        inputs["img"]
    )
    
    # Retrieve pre-loaded model from loader
    pipe = inputs["context"]

    torch.backends.cuda.matmul.allow_tf32 = True

    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                image=init_image,
                mask_image=mask_image,
                control_image=control_image
    ).images[0]

    print(f"Saved Image: {image}")
    image.save("output.png")