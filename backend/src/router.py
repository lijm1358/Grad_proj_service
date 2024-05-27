import os
from typing import Annotated

from database import get_session
from fastapi import APIRouter, Depends
from google.cloud.storage import Bucket
from model import LogImggen, LogImggenRequest, LogRecommendation, LogUserItemInteraction, User
from schemas import ImageGen, Interact, Recommend
from service import (
    article_id_to_info,
    generate_embedding_from_image,
    generate_image_from_prompt,
    predict_prompt_class,
    recommend_item_from_seqimg,
)
from sqlmodel import Session, select
from utils import gcp_storage_conn, gcp_storage_upload

api_router = APIRouter()


@api_router.post("/imagegen")
def generate_image(
    data: ImageGen,
    bucket: Annotated[Bucket, Depends(gcp_storage_conn)],
    db_session: Annotated[Session, Depends(get_session)],
):
    generated_img_paths = generate_image_from_prompt(data.user_id, data.prompt)
    generated_emb_paths = generate_embedding_from_image(generated_img_paths)

    dest_img_path_list = []
    dest_emb_path_list = []
    for img_path, emb_path in zip(generated_img_paths, generated_emb_paths):
        dest_img_path = "/".join(img_path.split("/")[1:])
        dest_emb_path = "/".join(emb_path.split("/")[1:])
        dest_img_path_list.append(dest_img_path)
        dest_emb_path_list.append(dest_emb_path)
        gcp_storage_upload(bucket, img_path, dest_img_path)
        gcp_storage_upload(bucket, emb_path, dest_emb_path)

    image_urls = [
        "https://storage.googleapis.com/" + os.environ["GCP_BUCKET_NAME"] + "/" + path for path in dest_img_path_list
    ]
    embedding_urls = [
        "https://storage.googleapis.com/" + os.environ["GCP_BUCKET_NAME"] + "/" + path for path in dest_emb_path_list
    ]
    image_ids = [url.split("/")[-1][:-4] for url in image_urls]

    requested_user = db_session.exec(select(User).where(User.username == data.user_id)).one()
    imggen_request = LogImggenRequest(user=requested_user, prompt=data.prompt)
    for image_url, embedding_url, image_id in zip(image_urls, embedding_urls, image_ids):
        new_image = LogImggen(
            id=image_id,
            image_location=image_url,
            emb_location=embedding_url,
            selected=False,
            request_log=imggen_request,
        )
        db_session.add(new_image)
    db_session.commit()

    return {
        "user_id": data.user_id,
        "prompt": data.prompt,
        "images": [
            {"image_id": image_id, "image_url": image_url} for image_id, image_url in zip(image_ids, image_urls)
        ],
    }


@api_router.post("/recommend")
def recommend_item(data: Recommend, db_session: Annotated[Session, Depends(get_session)]):
    gen_img_row = db_session.get(LogImggen, data.image_id)
    emb_url = gen_img_row.emb_location
    prompt_res = predict_prompt_class(data.prompt)

    user_db_id = db_session.exec(select(User).where(User.username == data.user_id)).one().id
    user_seq = db_session.exec(select(LogUserItemInteraction).where(LogUserItemInteraction.user_id == user_db_id))
    user_seq_ids = [seq.item_id for seq in user_seq]

    recommended_items_id, p_label = recommend_item_from_seqimg(user_seq_ids, data.image_id, emb_url, prompt_res)

    for rank, item_id in enumerate(recommended_items_id):
        item_id_original = "0" + item_id
        rec_request = LogRecommendation(
            item_rank=rank + 1, item_id=item_id_original, request_log_id=gen_img_row.request_log_id
        )
        db_session.add(rec_request)
    db_session.commit()

    rec_results = []
    for item_id in recommended_items_id:
        if len(rec_results) == 10:
            break
        article_info = article_id_to_info(item_id, p_label)
        if article_info:
            rec_results.append(article_info)

    return {
        "user_id": data.user_id,
        "image_id": data.image_id,
        "rec_results": rec_results,
        "related_request_id": gen_img_row.request_log_id,
    }


@api_router.post("/interact")
def item_interact(data: Interact, db_session: Annotated[Session, Depends(get_session)]):
    item_id_original = "0" + data.item_id
    interacted_user = db_session.exec(select(User).where(User.username == data.user_id)).one()

    related_imggen_request = db_session.get(LogImggenRequest, data.related_req_id)

    interaction_request = LogUserItemInteraction(
        user=interacted_user, item_id=item_id_original, request_log=related_imggen_request
    )
    db_session.add(interaction_request)
    db_session.commit()

    return {"message": "Item interaction recorded"}
