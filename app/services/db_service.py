# db_service.py
from sqlalchemy.orm import Session
from app.models.lab import Lab
from app.models.paper import Paper
from app.services.llm_service import generate_lab_summary, generate_paper_summary

def insert_lab_with_summary(
    db: Session,
    professor_name: str,
    lab_name: str,
    university: str = None,
    department: str = None,
    homepage_text: str = None,
    homepage_url: str = None,
    google_scholar_url: str = None,
    image_url: str = None,
    education_text: str = None,
):
    """Generate an English summary of a lab introduction and store it in MySQL"""
    summary = generate_lab_summary(homepage_text)

    lab = Lab(
        professor_name=professor_name,
        lab_name=lab_name,
        university=university,
        department=department,
        summary=summary,
        homepage_url=homepage_url,
        google_scholar_url=google_scholar_url,
        image_url=image_url, 
        education_text=education_text,
    )
    db.add(lab)
    db.commit()
    db.refresh(lab)
    return lab


def insert_paper_with_summary(
    db: Session,
    lab_id: int,
    title: str,
    abstract_text: str,
    publication_year: int = None,
    keywords=None,
    citation_count: int = 0,
    doi: str = None,
    google_scholar_url: str = None,
):
    """Generate an English summary of a research paper and store it in MySQL"""
    summary = generate_paper_summary(title, abstract_text)

    paper = Paper(
        lab_id=lab_id,
        title=title,
        abstract=abstract_text, 
        summary=summary,
        publication_year=publication_year,
        keywords=keywords,
        citation_count=citation_count,
        doi=doi,
        google_scholar_url=google_scholar_url,
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)
    return paper
