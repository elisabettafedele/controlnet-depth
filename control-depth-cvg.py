from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from transformers import pipeline
import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

checkpoint = "lllyasviel/control_v11f1p_sd15_depth"
image_file = 'room1_view3.png'

image = Image.open(f'inputs/{image_file}')

# prompt = "A picture of an indoor room." #"A picture of a pink living room." "A picture of a living room in a chalet."  "A picture of a living room ." "A picture of a living room in a scandinavian style."
"A corner of a room for kids." 
prompt = "A corner of a room with a plant." 
#prompt = "A picture of a pink living room."

depth_estimator = pipeline('depth-estimation')
image = depth_estimator(image)['depth']
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)
# control_image = Image.open("/home/efedele/Programming/thesis/LooseControl/assets/office0_depth.png")

#control_image.save("/home/efedele/Programming/thesis/LooseControl/images_iccv/depth_room0_pink.png")

controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

image_name = image_file.split('.')[0]

image.save(f'outputs/{image_name}.png')