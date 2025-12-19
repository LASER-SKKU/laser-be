from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.lab_recommendation_service import recommend_labs

router = APIRouter(prefix="/labs", tags=["Labs Recommendation"])

class RecommendRequest(BaseModel):
    user_text: str
    similarity_threshold: float = 0.5
    top_k: int = 5


@router.post("/recommend")
def recommend_labs_endpoint(
    req: RecommendRequest = Body(...),
    db: Session = Depends(get_db),
):
    """
    연구실 추천 API
    - 입력(JSON):
      {
        "user_text": "자연어 입력",
        "similarity_threshold": 0.7,
        "top_k": 10
      }
    """
    try:
        results = recommend_labs(
            db=db,
            user_text=req.user_text,
            similarity_threshold=req.similarity_threshold,
            top_k=req.top_k,
        )

        if not results:
            return {
                "message": "추천할 연구실이 없습니다. (유사도 기준 미달)",
                "results": [],
            }

        ranked_results = [{"rank": i + 1, **lab} for i, lab in enumerate(results)]

        return {
            "message": "추천 완료",
            "criteria": {
                "similarity_threshold": req.similarity_threshold,
                "top_k": req.top_k,
            },
            "results": ranked_results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {e}")
