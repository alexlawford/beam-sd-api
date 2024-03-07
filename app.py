from beam import App, Runtime, Image, Output, Volume
import os
import torch
import PIL
import numpy as np
from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionLatentUpscalePipeline, AutoPipelineForImage2Image, ControlNetModel
from diffusers.utils import load_image
import base64
from io import BytesIO

cache_path = "./models"

# Environment app runs on
app = App(
    name="stable-diffusion-app",
    runtime=Runtime(
        cpu=1,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.8",
            python_packages=[
                "diffusers[torch]>=0.26.3",
                "transformers",
                "torch",
                "pillow",
                "numpy",
                "accelerate",
                "safetensors",
                "xformers",
                "compel"
            ],
        ),
    ),
    volumes=[Volume(name="models", path="./models")],
)

app.rest_api(keep_warm_seconds=600)

def decode_base64_image(image_string):
    image_string = image_string[len("data:image/png;base64,"):]
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    image = PIL.Image.open(buffer)
    rgb = image.convert('RGB')
    return rgb

# Runs once when the container first boots
def load_models():
    torch.backends.cuda.matmul.allow_tf32 = True

    scribble = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16
    )

    openpose = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16
    )

    inpaintScribble = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=scribble,
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    inpaintOpenpose = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=openpose,
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    upscale = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    refine = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    inpaintScribble.enable_xformers_memory_efficient_attention()
    inpaintOpenpose.enable_xformers_memory_efficient_attention()
    upscale.enable_xformers_memory_efficient_attention()
    refine.enable_xformers_memory_efficient_attention()

    return (inpaintScribble, inpaintOpenpose, upscale, refine)

@app.task_queue(
    loader=load_models,
    outputs=[Output(path="output.png")],
)
def generate_image(**inputs):
        
    img = load_image(
        "./plain-background.png"
    )

    generator = torch.Generator(device="cuda").manual_seed(0)
    
    # Retrieve pre-loaded models from loader
    (inpaintScribble, inpaintOpenpose, upscale, refine) = inputs["context"]

    layers = inputs["layers"]

    full_prompt = ""

    for layer in layers:

        mask_image = decode_base64_image(
            layer["mask"]
        )

        control_image = decode_base64_image(
            layer["control"]
        )

        prompt = layer["prompt"]

        full_prompt = full_prompt + ' ' + prompt

        if layer["type"] == "figure":
            img = inpaintOpenpose(
                prompt=prompt,
                image=img,
                mask_image=mask_image,
                control_image=control_image,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=generator,
                controlnet_conditioning_scale=0.75
            ).images[0]
        else:
            img = inpaintScribble(
                prompt=prompt,
                image=img,
                mask_image=mask_image,
                control_image=control_image,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=generator,
                controlnet_conditioning_scale=0.75
            ).images[0]

    upscaled = upscale(
        prompt=full_prompt,
        image=img,
        num_inference_steps=20,
        guidance_scale=6.0,
        generator=generator,
    ).images[0]

    refined = refine(
        prompt=full_prompt,
        image=upscaled,
        num_inference_steps=50,
        guidance_scale=6.0,
        generator=generator,
        strength=0.25,
    ).images[0]

    print(f"Saved Image: {refined}")
    refined.save("output.png")