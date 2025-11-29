# app/api/routes/paper_embedding_route.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.paper_embedding_service import (
    insert_paper_embeddings,
    get_paper_collection_count,
    drop_paper_collection
)

router = APIRouter(prefix="/papers", tags=["Papers Embedding"])

@router.post("/embed")
def embed_papers(db: Session = Depends(get_db)):
    try:
        return insert_paper_embeddings(db)
    except Exception as e:
        raise HTTPException(500, f"Paper embedding error: {e}")


@router.get("/embed/count")
def count_papers():
    try:
        return get_paper_collection_count()
    except Exception as e:
        raise HTTPException(500, f"Count error: {e}")


@router.delete("/embed")
def delete_papers():
    try:
        return drop_paper_collection()
    except Exception as e:
        raise HTTPException(500, f"Delete error: {e}")
