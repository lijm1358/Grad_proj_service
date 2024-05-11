from pydantic import BaseModel

# from models import Event


class ImageGen(BaseModel):
    user_id: str
    prompt: str

    class Config:
        json_schema_extra = {
            "example": {"user_id": "1234", "prompt": "A white t-shirt with a blue collar and a red stripe"}
        }


class Recommend(BaseModel):
    user_id: str
    image_id: str

    class Config:
        json_schema_extra = {"example": {"user_id": "1234", "image_id": "id1"}}


class Interact(BaseModel):
    user_id: str
    item_id: str

    class Config:
        json_schema_extra = {"example": {"user_id": "1234", "item_id": "0108775015"}}
