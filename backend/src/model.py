from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    username: str = Field(unique=True)
    password: str

    imggen_requests: List["LogImggenRequest"] = Relationship(back_populates="user")
    interactions: List["LogUserItemInteraction"] = Relationship(back_populates="user")


class Item(SQLModel, table=True):
    article_id: str = Field(primary_key=True, unique=True)
    product_code: str
    prod_name: str
    product_type_no: int
    product_type_name: str
    product_group_name: str
    graphical_appearance_no: int
    graphical_appearance_name: str
    colour_group_code: str
    colour_group_name: str
    perceived_colour_value_id: int
    perceived_colour_value_name: str
    perceived_colour_master_id: int
    perceived_colour_master_name: str
    department_no: int
    department_name: str
    index_code: str
    index_name: str
    index_group_no: int
    index_group_name: str
    section_no: int
    section_name: str
    garment_group_no: int
    garment_group_name: str
    detail_desc: str

    recommendations: List["LogRecommendation"] = Relationship(back_populates="item")
    interactions: List["LogUserItemInteraction"] = Relationship(back_populates="item")


class LogImggenRequest(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    prompt: str
    timestamp: datetime = Field(default_factory=datetime.now)

    user: "User" = Relationship(back_populates="imggen_requests")
    images: List["LogImggen"] = Relationship(back_populates="request_log")
    recommendations: List["LogRecommendation"] = Relationship(back_populates="request_log")


class LogImggen(SQLModel, table=True):
    id: str = Field(primary_key=True)
    image_location: str
    emb_location: str
    selected: bool
    request_log_id: int = Field(foreign_key="logimggenrequest.id")

    request_log: "LogImggenRequest" = Relationship(back_populates="images")


class LogRecommendation(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    item_rank: int = Field(ge=1, le=10)
    item_id: str = Field(foreign_key="item.article_id")
    request_log_id: int = Field(foreign_key="logimggenrequest.id")

    item: "Item" = Relationship(back_populates="recommendations")
    request_log: "LogImggenRequest" = Relationship(back_populates="recommendations")
    interactions: Optional["LogUserItemInteraction"] = Relationship(back_populates="recommendation_log")


class LogUserItemInteraction(SQLModel, table=True):
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    item_id: str = Field(foreign_key="item.article_id")
    recommendation_log_id: Optional[int] = Field(default=None, foreign_key="logrecommendation.id")
    timestamp: datetime = Field(default_factory=datetime.now, primary_key=True)

    user: "User" = Relationship(back_populates="interactions")
    item: "Item" = Relationship(back_populates="interactions")
    recommendation_log: Optional["LogRecommendation"] = Relationship(back_populates="interactions")
