import os
import uuid

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPModel
from utils import get_current_date_str

BATCH_SIZE = 4
WIDTH = 512
HEIGHT = 512
INFERENCE_STEP = 30

gen_img_save_path = "../gen_images"


def generate_image_from_prompt(user_id: int, prompt: str) -> None:
    filename = get_current_date_str()

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

    os.makedirs(os.path.join(gen_img_save_path, filename, str(user_id)), exist_ok=True)
    for _, img in enumerate(images_list):
        id = uuid.uuid4().hex
        img.save(os.path.join(gen_img_save_path, filename, str(user_id), f"{id}.jpg"))
