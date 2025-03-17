from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image

checkpoint = "lllyasviel/control_v11f1p_sd15_depth"

depth_estimator = pipeline('depth-estimation')

image_file = 'room1_view3.png'

image = Image.open(f'inputs/{image_file}')

image = depth_estimator(image)['depth']
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    checkpoint, torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
) # runwayml/stable-diffusion-v1-5 or dreamlike-art/dreamlike-photoreal-2.0

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

image = pipe("A corner of a room with a plant.", image, num_inference_steps=30).images[0]
image_name = image_file.split('.')[0]

image.save(f'outputs/{image_name}.png')