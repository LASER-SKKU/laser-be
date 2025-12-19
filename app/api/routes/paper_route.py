# app/api/routes/paper_route.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, selectinload
from pydantic import BaseModel, Field
from typing import List, Optional

from app.core.database import get_db
from app.models.paper import Paper
from app.models.lab import Lab

router = APIRouter(prefix="/papers", tags=["Papers"])

class PaperSummary(BaseModel):
    """목록 응답용(가벼운) 스키마"""
    paper_id: int
    title: str
    publication_year: Optional[int] = None
    citation_count: int = 0

    model_config = {"from_attributes": True}  # Pydantic v2


class PaperDetail(BaseModel):
    """상세 응답용 스키마"""
    paper_id: int
    lab_id: int
    title: str
    abstract: Optional[str] = None
    summary: Optional[str] = None
    publication_year: Optional[int] = None
    keywords: Optional[dict] = None
    citation_count: int = 0
    doi: Optional[str] = None
    google_scholar_url: Optional[str] = None

    model_config = {"from_attributes": True}  # Pydantic v2


@router.get("/{paper_id}", response_model=PaperDetail, status_code=status.HTTP_200_OK)
def get_paper_detail(paper_id: int, db: Session = Depends(get_db)):
    """
    paper_id로 단일 논문 상세 정보를 반환합니다.
    """
    paper: Optional[Paper] = (
        db.query(Paper)
        .options(selectinload(Paper.lab))  # 필요 시 연구실 참조 로딩(응답엔 포함하지 않음)
        .filter(Paper.paper_id == paper_id)
        .first()
    )

    if not paper:
        raise HTTPException(status_code=404, detail="해당 paper_id의 논문을 찾을 수 없습니다.")

    return paper


@router.get("", response_model=List[PaperSummary], status_code=status.HTTP_200_OK)
def list_papers_by_lab(
    lab_id: int = Query(..., description="검색할 연구실 ID"),
    limit: int = Query(50, ge=1, le=200, description="한 번에 가져올 개수"),
    offset: int = Query(0, ge=0, description="건너뛸 개수(페이지네이션)"),
    db: Session = Depends(get_db),
):
    """
    lab_id(필수)로 논문 목록을 조회합니다. (경량 응답)
    - PaperSummary 스키마로 가볍게 반환
    - 기본 정렬: publication_year DESC, citation_count DESC, title ASC
    """
    query = db.query(Paper).filter(Paper.lab_id == lab_id)

    papers: List[Paper] = (
        query.order_by(
            Paper.publication_year.is_(None), Paper.publication_year.desc(),
            Paper.citation_count.desc(),
            Paper.title.asc(),
        )
        .offset(offset)
        .limit(limit)
        .all()
    )

    if not papers:
        raise HTTPException(status_code=404, detail="조건에 맞는 논문을 찾을 수 없습니다.")

    return papers
