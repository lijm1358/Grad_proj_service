import os
import posixpath
import uuid
from random import sample
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from dotenv import load_dotenv
from fashion_clip.fashion_clip import FashionCLIP
from models.mlpbert import MLPBERT4Rec
from transformers import CLIPModel
from utils import get_current_date_str, load_json

BATCH_SIZE = 4
WIDTH = 512
HEIGHT = 512
INFERENCE_STEP = 30

load_dotenv("../.env")

gen_img_save_path = os.environ["GEN_IMG_SAVE_PATH"]
gen_emb_save_path = os.environ["GEN_EMB_SAVE_PATH"]
setting_path = os.environ["RECOMMENDER_MODEL_SETTING_PATH"]

device = "cuda:0"

with open(setting_path) as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

path = "../data"
metadata = load_json(f"{path}/metadata.json")
test_data = torch.load(f"{path}/test_data.pt")  # item sequence per user
item2idx = torch.load(f"{path}/item2idx.pt")  # mapping test_data's item idx to article id.
idx2item = {k: v for v, k in item2idx.items()}
idx_groups = torch.load(f"{path}/id_group_dict.pt")
gen_img_emb = torch.load(f"{path}/gen_img_emb.pt")
text_emb = torch.load(f"{path}/detail_text_embeddings.pt")

num_user = metadata["num of user"]
num_item = metadata["num of item"]

model_name: str = settings["model_name"]
model_args: dict = settings["model_arguments"]

if model_name in ("MLPBERT4Rec", "MLPRec", "MLPwithBERTFreeze"):
    model_args["gen_img_emb"] = gen_img_emb
    model_args["text_emb"] = text_emb
else:
    model_args["gen_img_emb"], model_args["text_emb"] = None, None

if model_args["gen_img_emb"] is not None:
    model_args["linear_in_size"] = model_args["gen_img_emb"].shape[-1] * model_args["num_gen_img"]
if model_args["text_emb"] is not None:
    model_args["linear_in_size"] = model_args["text_emb"].shape[-1]

articles = pd.read_csv(os.environ["ITEM_INFO_PATH"])


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


def __convert_imgpath_to_embpath(path: str) -> str:
    date_dir = get_current_date_str()

    filename = path.split("/")[-1]
    filename = filename[:-4] + ".pth"
    emb_path_dir = posixpath.join(gen_emb_save_path, date_dir)
    os.makedirs(emb_path_dir, exist_ok=True)
    emb_path = posixpath.join(emb_path_dir, filename)

    return emb_path


def generate_embedding_from_image(path_list: List[str]):
    fclip = FashionCLIP("fashion-clip")

    image_embeddings = fclip.encode_images(path_list, batch_size=4)

    for image_embedding, path in zip(image_embeddings, path_list):
        torch.save(image_embedding, __convert_imgpath_to_embpath(path))


def __get_modal_emb(tokens):
    item_ids = tokens.clone().detach()
    item_ids = item_ids[:-1]
    item_ids -= 1
    modal_emb = torch.tensor([])

    if idx_groups is not None:
        item_id_group_sampler = np.vectorize(lambda x: sample(idx_groups[x], k=1)[0] if x != -1 else -1)
        item_ids = item_id_group_sampler(item_ids)

    if gen_img_emb is not None:
        img_idx = sample([0, 1, 2], k=1)
        modal_emb = torch.flatten(gen_img_emb[item_ids][:, img_idx, :], start_dim=-2, end_dim=-1)
        if model_args["img_noise"]:
            modal_emb += torch.randn_like(modal_emb) * model_args["std"] + model_args["mean"]  # add noise to gen image

    # if self.text_emb is not None:
    #     modal_emb = self.text_emb[item_ids]  # detail text embedding

    return modal_emb


def __select_data(idx, additional_emb):
    # TODO: change idx retrieve method if idx can be other than integer
    idx = int(idx)
    tokens = torch.tensor(test_data[idx] + [num_item], dtype=torch.long) + 1
    tokens = tokens[-model_args["max_len"] :]
    mask_len = model_args["max_len"] - len(tokens)
    tokens = torch.concat((torch.zeros(mask_len, dtype=torch.long), tokens), dim=0)

    img_emb = __get_modal_emb(tokens)
    img_emb = torch.concat((img_emb, torch.tensor(additional_emb).unsqueeze(0)), dim=0)

    return tokens, img_emb


# TODO: load embeddings not from image_id, load from database information about embedding path.
def __load_img_embedding(image_id: str) -> torch.Tensor:
    path = posixpath.join(os.environ["GEN_EMBEDDING_LOAD_PATH"], image_id + ".pth")
    emb = torch.load(path)

    return emb


def recommend_item_from_seqimg(user_idx: int, image_id: str):
    model = MLPBERT4Rec(
        **model_args,
        num_item=num_item,
        idx_groups=idx_groups,
        device=device,
    ).to(device)

    model_ckpt_path = os.environ["RECOMMENDER_MODEL_CKPT_PATH"]
    model.load_state_dict(torch.load(model_ckpt_path))

    model.eval()

    chosen_item_emb = __load_img_embedding(image_id)

    tokens, modal_emb = __select_data(user_idx, chosen_item_emb)
    tokens = tokens.unsqueeze(0).to(device)
    modal_emb = modal_emb.unsqueeze(0).to(device)
    labels = [0] * (model_args["max_len"] - 1) + [1]
    labels = torch.tensor(labels).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(log_seqs=tokens, modal_emb=modal_emb, labels=labels)

    user_res = -logits[0, -1, 1:]
    recommended_idx = user_res.argsort()[:10].tolist()
    recommended_idx = [str(idx2item[idx]) for idx in recommended_idx]

    return recommended_idx


def article_id_to_info(article_id):
    real_id = "0" + article_id
    articles["article_id"]
    single_item = articles[articles["article_id"] == int(article_id)].squeeze()
    base_img_url = os.environ["GCP_ORIG_IMAGE_URL"]
    base_img_url += real_id[:3] + "/" + real_id + ".jpg"

    infos = {
        "article_id": str(single_item["article_id"]),
        "prod_name": single_item["prod_name"],
        "prod_type_name": single_item["product_type_name"],
        "detail_desc": single_item["detail_desc"],
        "image_link": base_img_url,
    }

    return infos
