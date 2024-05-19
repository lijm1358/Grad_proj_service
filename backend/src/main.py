from dotenv import load_dotenv

load_dotenv("../.env")

import uvicorn  # noqa: E402
from database import conn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from router import api_router  # noqa: E402

app = FastAPI()

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
def on_startup():
    conn()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
