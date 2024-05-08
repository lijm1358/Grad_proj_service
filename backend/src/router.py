import os
from typing import Annotated

from fastapi import APIRouter, Depends
from google.cloud.storage import Bucket
from schemas import ImageGen, Interact, Recommend
from service import (
    article_id_to_info,
    generate_embedding_from_image,
    generate_image_from_prompt,
    recommend_item_from_seqimg,
)
from utils import gcp_storage_conn, gcp_storage_upload

api_router = APIRouter()


@api_router.post("/imagegen")
def generate_image(data: ImageGen, bucket: Annotated[Bucket, Depends(gcp_storage_conn)]):
    generated_img_paths = generate_image_from_prompt(data.user_id, data.prompt)
    generate_embedding_from_image(generated_img_paths)

    dest_path_list = []
    for path in generated_img_paths:
        dest_path = "/".join(path.split("/")[1:])
        dest_path_list.append(dest_path)
        gcp_storage_upload(bucket, path, dest_path)

    image_urls = [
        "https://storage.googleapis.com/" + os.environ["GCP_BUCKET_NAME"] + "/" + path for path in dest_path_list
    ]
    image_ids = [url.split("/")[-1][:-4] for url in image_urls]

    return {
        "user_id": data.user_id,
        "prompt": data.prompt,
        "images": [
            {"image_id": image_id, "image_url": image_url} for image_id, image_url in zip(image_urls, image_ids)
        ],
    }


@api_router.post("/recommend")
def recommend_item(data: Recommend):
    recommended_items_id = recommend_item_from_seqimg(data.user_id, data.image_id)

    rec_results = [article_id_to_info(item_id) for item_id in recommended_items_id]

    return {
        "user_id": data.user_id,
        "image_id": data.image_id,
        "rec_results": rec_results,
    }


@api_router.post("/interact")
def item_interact(data: Interact):
    print(data)
    return {"message": "Item interaction recorded"}
