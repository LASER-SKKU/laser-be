# app/api/routes/lab_embedding_route.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.lab_embedding_service import (
    insert_lab_embeddings,
    get_lab_collection_count,
    drop_lab_collection,
)

router = APIRouter(prefix="/labs", tags=["Labs Embedding"])


@router.post("/embed")
async def embed_labs_into_milvus(db: Session = Depends(get_db)):
    """
    연구실 summary를 Milvus에 임베딩으로 저장.
    - 컬렉션이 없으면 자동 생성
    - 컬렉션이 있으면 유지 (덮어쓰기 없음)
    """
    try:
        result = insert_lab_embeddings(db)
        return {
            "message": "Processed lab embeddings.",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus embedding error: {e}")


@router.get("/embed/count")
async def count_lab_embeddings():
    """
    lab_embeddings 컬렉션 엔트리 수 반환.
    컬렉션이 없으면 count=0
    """
    try:
        result = get_lab_collection_count()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus count error: {e}")


@router.delete("/embed")
async def delete_lab_embeddings_collection():
    """
    lab_embeddings 컬렉션 삭제 (존재할 때만).
    """
    try:
        result = drop_lab_collection()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus drop error: {e}")
