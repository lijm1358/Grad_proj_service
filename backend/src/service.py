import os
import posixpath
import uuid
from typing import List

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPModel
from utils import get_current_date_str

BATCH_SIZE = 4
WIDTH = 512
HEIGHT = 512
INFERENCE_STEP = 30

gen_img_save_path = "../gen_images"


def generate_image_from_prompt(user_id: int, prompt: str) -> List[str]:
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        custom_pipeline="clip_guided_stable_diffusion",
        torch_dtype=torch.float16,
        clip_model=clip_model,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:0")

    images_list = []
    for _ in range(3):
        images = pipe(prompt, width=WIDTH, height=HEIGHT, num_inference_steps=INFERENCE_STEP).images[0]
        images_list.append(images)

    date_dir = get_current_date_str()
    img_path_dir = posixpath.join(gen_img_save_path, date_dir, str(user_id))
    img_path_list = []
    os.makedirs(img_path_dir, exist_ok=True)
    for _, img in enumerate(images_list):
        img_id = uuid.uuid4().hex

        img_path = posixpath.join(img_path_dir, f"{img_id}.jpg")
        img.save(img_path)
        img_path_list.append(img_path)

    return img_path_list
