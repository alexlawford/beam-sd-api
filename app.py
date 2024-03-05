from beam import App, Runtime, Image, Output, Volume
import os
import torch
import PIL
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel
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

    controlnetXl = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",
        torch_dtype=torch.float16
    )

    inPaintPipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    sdxlImg2ImgPipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        controlnet=controlnetXl,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=cache_path,     
    ).to("cuda")

    inPaintPipe.enable_xformers_memory_efficient_attention()
    sdxlImg2ImgPipe.enable_xformers_memory_efficient_attention()

    return (inPaintPipe, sdxlImg2ImgPipe)

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

    generator = torch.Generator(device="cuda").manual_seed(0)

    prompt = "A man holding a camera in the artic"
    
    # Retrieve pre-loaded models from loader
    (inPaintPipe, sdxlImg2ImgPipe) = inputs["context"]

    image = inPaintPipe(
        width=1024 / 2,
        height=576 / 2,
        prompt=prompt,
        image=img,
        mask_image=mask_image,
        control_image=control_image,
        guidance_scale=8.0,
        num_inference_steps=50,
        generator=generator
    ).images[0]

    image = image.resize((1024, 576))
    control_image = image.resize((1024, 576))

    image = sdxlImg2ImgPipe(
        width=1024,
        height=576,
        prompt=prompt,
        image=image,
        control_image=control_image,
        guidance_scale=8.0,
        num_inference_steps=100,
        strength=0.25,
        generator=generator,
    ).images[0]

    print(f"Saved Image: {image}")
    image.save("output.png")