# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import Base, engine
import app.models.lab 
import app.models.paper
from app.api.routes import (
    lab_loader_route,
    lab_embedding_route,
    lab_recommendation_route,
    lab_route,
    paper_loader_route,
    paper_route,
    paper_embedding_route
)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="LASER Backend API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 라우터 등록
app.include_router(lab_loader_route.router)
app.include_router(lab_embedding_route.router)
app.include_router(lab_recommendation_route.router)
app.include_router(lab_route.router)
app.include_router(paper_loader_route.router)
app.include_router(paper_route.router)
app.include_router(paper_embedding_route.router)


@app.get("/")
def root():
    return {"message": "Welcome to LASER Backend!"}
