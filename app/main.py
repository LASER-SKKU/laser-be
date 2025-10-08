from fastapi import FastAPI
from app.core.database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Laser Backend API")

@app.get("/")
def root():
    return {"message": "Welcome to Laser Backend!"}
