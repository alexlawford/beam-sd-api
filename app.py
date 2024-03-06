from beam import App, Runtime, Image, Output, Volume
import os
import torch
import PIL
from diffusers import AutoPipelineForInpainting, StableDiffusionLatentUpscalePipeline, AutoPipelineForImage2Image, ControlNetModel
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
                "diffusers[torch]>=0.26.3",
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

app.rest_api(keep_warm_seconds=0)

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

mask_image_2 = load_image(
    "./example_mask_02.png"
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

    inpaint = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    inpaintnet = AutoPipelineForInpainting.from_pipe(
        inpaint,
        controlnet=controlnet,
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

    inpaint.enable_xformers_memory_efficient_attention()
    inpaintnet.enable_xformers_memory_efficient_attention()
    upscale.enable_xformers_memory_efficient_attention()
    refine.enable_xformers_memory_efficient_attention()

    return (inpaint, inpaintnet, upscale, refine)

@app.task_queue(
    loader=load_models,
    outputs=[Output(path="output.png")],
)
def generate_image(**inputs):
    
    prompt = inputs["prompt"]
    seed = inputs["seed"]

    # mask_image = decode_base64_image(
    #     inputs["mask_image"]
    # )

    # control_image = decode_base64_image(
    #     inputs["control_image"]
    # )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Retrieve pre-loaded models from loader
    (inpaint, inpaintnet, upscale, refine) = inputs["context"]

    latents = inpaintnet(
        prompt=prompt,
        image=img,
        mask_image=mask_image,
        control_image=control_image,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        controlnet_conditioning_scale=0.75,
    ).images[0]

    latents = inpaint(
        prompt='the moon, very large',
        image=latents,
        mask_image=mask_image_2,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        controlnet_conditioning_scale=0.75,
        output_type="latent",
    ).images

    upscaled = upscale(
        prompt=prompt + ', the moon, very large',
        image=latents,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    image = refine(
        prompt=prompt + ', the moon, very large',
        image=upscaled,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        strength=0.25,
    ).images[0]

    print(f"Saved Image: {image}")
    image.save("output.png")