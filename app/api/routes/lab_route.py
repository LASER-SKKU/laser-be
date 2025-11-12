# app/api/routes/lab_route.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, selectinload
from pydantic import BaseModel, Field
from typing import List, Optional

from app.core.database import get_db
from app.models.lab import Lab
from app.models.paper import Paper

router = APIRouter(prefix="/labs", tags=["Labs"])

# ---------- Pydantic Schemas ----------
class PaperInLab(BaseModel):
    paper_id: int
    title: str
    publication_year: Optional[int] = None
    citation_count: int = 0

    model_config = {"from_attributes": True}  # Pydantic v2


class LabSummary(BaseModel):
    """목록 응답용(가벼운) 스키마: 불필요한 필드는 제외하고 논문도 포함하지 않음"""
    lab_id: int
    professor_name: str
    university: Optional[str] = None
    department: Optional[str] = None
    summary: Optional[str] = None

    model_config = {"from_attributes": True}  # Pydantic v2


class LabDetail(BaseModel):
    """상세 응답용 스키마: 논문 목록 포함"""
    lab_id: int
    professor_name: str
    university: Optional[str] = None
    department: Optional[str] = None
    summary: Optional[str] = None
    homepage_url: Optional[str] = None
    google_scholar_url: Optional[str] = None
    papers: List[PaperInLab] = Field(default_factory=list)

    model_config = {"from_attributes": True}  # Pydantic v2


# ------------------------------ Routes ------------------------------
@router.get("/{lab_id}", response_model=LabDetail, status_code=status.HTTP_200_OK)
def get_lab_detail(lab_id: int, db: Session = Depends(get_db)):
    """
    lab_id로 단일 연구실 상세 정보를 반환합니다.
    - 논문 목록까지 포함(selectinload)
    """
    lab: Optional[Lab] = (
        db.query(Lab)
        .options(selectinload(Lab.papers))
        .filter(Lab.lab_id == lab_id)
        .first()
    )

    if not lab:
        raise HTTPException(status_code=404, detail="해당 lab_id의 연구실을 찾을 수 없습니다.")

    return lab


@router.get("", response_model=List[LabSummary], status_code=status.HTTP_200_OK)
def list_labs_by_department(
    department: str = Query(..., min_length=1, description="검색할 학과명(부분 일치)"),
    university: Optional[str] = Query(None, description="(선택) 대학명으로 추가 필터"),
    limit: int = Query(50, ge=1, le=200, description="한 번에 가져올 개수"),
    offset: int = Query(0, ge=0, description="건너뛸 개수(페이지네이션)"),
    db: Session = Depends(get_db),
):
    """
    department(필수)로 연구실 목록을 조회합니다. (경량 응답)
    - 부분 일치(ilike) 검색
    - university가 주어지면 추가 필터
    - 논문 비포함(LabSummary)으로 가볍게 반환
    - MySQL 호환 정렬: NULL이 마지막으로 가도록 is_(None) + 실제 값 오름차순
    """
    query = (
        db.query(Lab)
        # 목록은 가볍게: papers 미로딩
        .filter(Lab.department.ilike(f"%{department}%"))
    )

    if university:
        query = query.filter(Lab.university.ilike(f"%{university}%"))

    labs: List[Lab] = (
        query.order_by(
            Lab.university.is_(None), Lab.university.asc(),
            Lab.department.is_(None), Lab.department.asc(),
            Lab.professor_name.asc(),
        )
        .offset(offset)
        .limit(limit)
        .all()
    )

    if not labs:
        raise HTTPException(status_code=404, detail="조건에 맞는 연구실을 찾을 수 없습니다.")

    return labs
