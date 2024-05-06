import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from router import api_router

# from database import conn

app = FastAPI()

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
def on_startup():
    load_dotenv("../.env")
    # conn()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
