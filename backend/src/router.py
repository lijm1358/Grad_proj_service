import os
from typing import Annotated

from fastapi import APIRouter, Depends
from google.cloud.storage import Bucket
from schemas import ImageGen, Interact, Recommend
from service import generate_embedding_from_image, generate_image_from_prompt
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
    print(data)
    return {
        "user_id": data.user_id,
        "image_id": data.image_id,
        "rec_results": [
            {
                "item_id": "0108775015",
                "prod_name": "Strap top",
                "prod_type_name": "Vest top",
                "detail_desc": "Womens Everyday Basics,1002,Jersey Basic,Jersey top with narrow shoulder straps.",
                "image_link": "http://some.s3.link/0108775015.jpg",
            },
            {
                "item_id": "0160442042",
                "prod_name": "Sneaker 3p Socks",
                "prod_type_name": "Socks",
                "detail_desc": "Short, fine-knit socks designed to be hidden by your shoes with a silicone trim \
at the back of the heel to keep them in place.",
                "image_link": "http://some.s3.link/0160442042.jpg",
            },
        ],
    }


@api_router.post("/interact")
def item_interact(data: Interact):
    print(data)
    return {"message": "Item interaction recorded"}
