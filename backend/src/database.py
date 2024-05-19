import os

from sqlmodel import Session, SQLModel, create_engine

DB_URL = os.environ["DB_URL"]

engine_url = create_engine(DB_URL, echo=True)


def conn():
    SQLModel.metadata.create_all(engine_url)


def get_session():
    with Session(engine_url) as session:
        yield session
