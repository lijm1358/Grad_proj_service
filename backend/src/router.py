from fastapi import APIRouter
from schemas import ImageGen, Interact, Recommend
from service import generate_image_from_prompt

api_router = APIRouter()


@api_router.post("/imagegen")
def generate_image(data: ImageGen):
    generate_image_from_prompt(data.user_id, data.prompt)

    return {
        "user_id": data.user_id,
        "prompt": data.prompt,
        "images": [
            {
                "image_id": "id1",
                "image_url": "http://some.s3.link/id1.jpg",
            },
            {
                "image_id": "id2",
                "image_url": "http://some.s3.link/id2.jpg",
            },
            {
                "image_id": "id3",
                "image_url": "http://some.s3.link/id3.jpg",
            },
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
