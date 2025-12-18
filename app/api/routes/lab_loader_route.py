from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
from app.core.database import get_db
from app.services.lab_loader_service import load_lab_json_to_mysql

router = APIRouter(prefix="/labs", tags=["Labs"])


@router.post("/load")
async def load_lab_data_from_default_file(db: Session = Depends(get_db)):
    """
    app/data/labs_skkuniv_2025.jsonl 파일을 자동으로 읽어
    연구실 데이터를 MySQL에 저장합니다.
    """

    file_path = Path("app/data/labs_skkuniv_2025.jsonl")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file_path}")

    try:
        result = load_lab_json_to_mysql(db, str(file_path))
        return {
            "message": "연구실 데이터 로드 및 저장 완료",
            "file": file_path.name,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 저장 중 오류 발생: {e}")
