# app/api/routes/paper_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path

from app.core.database import get_db
from app.services.paper_loader_service import load_papers_json_to_mysql

router = APIRouter(prefix="/papers", tags=["Papers"])


@router.post("/load")
async def load_paper_data_from_default_file(db: Session = Depends(get_db)):
    """
    app/data/papers_skkuniv_2025.json 파일을 자동으로 읽어
    논문 데이터를 MySQL에 저장합니다.
    - (professor_name, university, department)를 기반으로 lab_id 매칭
    - lab_id 또는 title 누락 시 스킵 처리
    """
    file_path = Path("app/data/papers_skkuniv_2025.json")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file_path}")

    try:
        result = load_papers_json_to_mysql(db, str(file_path))
        return {
            "message": "논문 데이터 로드 및 저장 완료",
            "file": file_path.name,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 저장 중 오류 발생: {e}")
