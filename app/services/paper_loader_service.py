# app/services/paper_loader_service.py
import json
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.lab import Lab
from app.services.db_service import insert_paper_with_summary


def _norm(s: Optional[str]) -> Optional[str]:
    """
    간단한 정규화: 앞뒤 공백 제거.
    (대소문자 무시 매칭은 DB 레벨에서 func.lower로 처리)
    """
    if s is None:
        return None
    return s.strip()


def _find_lab_id(
    db: Session,
    professor_name: Optional[str],
    university: Optional[str],
    department: Optional[str],
    item_idx: int,
) -> Optional[int]:
    """
    (professor_name, university, department)로 labs에서 lab_id를 찾는다.
    1) 정확 매칭 시도
    2) 없으면 대소문자 무시 매칭 시도
    - 0건 또는 2건 이상일 때는 None 반환 (스킵)
    """
    p = _norm(professor_name)
    u = _norm(university)
    d = _norm(department)

    # 필수 키가 비어 있으면 찾을 수 없음
    if not p or not u or not d:
        print(f"[아이템 {item_idx}] lab 매칭 키 누락 → (professor_name, university, department) 필요")
        return None

    # 1) 정확 매칭
    q = (
        db.query(Lab.lab_id)
        .filter(
            Lab.professor_name == p,
            Lab.university == u,
            Lab.department == d,
        )
    )
    rows = q.all()
    if len(rows) == 1:
        return rows[0][0]
    elif len(rows) > 1:
        print(f"[아이템 {item_idx}] lab 매칭 다중 결과(정확 매칭) → 스킵 ({p}, {u}, {d})")
        return None

    # 2) 대소문자 무시 매칭
    q2 = (
        db.query(Lab.lab_id)
        .filter(
            func.lower(Lab.professor_name) == func.lower(p),
            func.lower(Lab.university) == func.lower(u),
            func.lower(Lab.department) == func.lower(d),
        )
    )
    rows2 = q2.all()
    if len(rows2) == 1:
        return rows2[0][0]
    elif len(rows2) > 1:
        print(f"[아이템 {item_idx}] lab 매칭 다중 결과(CI 매칭) → 스킵 ({p}, {u}, {d})")
        return None

    print(f"[아이템 {item_idx}] lab 미발견 → 스킵 ({p}, {u}, {d})")
    return None


def load_papers_json_to_mysql(db: Session, file_path: str) -> dict:
    """
    논문 JSON 파일(배열)을 읽어서 MySQL에 삽입하는 함수.
    - lab_id는 (professor_name, university, department)로 labs 테이블에서 조회
    - lab_id 또는 title이 없으면 저장하지 않고 다음 논문으로 넘어감
    - 요약(summary) 생성은 insert_paper_with_summary() 내부에서 처리
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"❌ 논문 데이터 파일을 찾을 수 없습니다: {file_path}")

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 루트가 배열이 아닙니다.")
    except Exception as e:
        raise ValueError(f"❌ JSON 파싱 오류: {e}")

    inserted_count = 0
    skipped_count = 0

    for idx, item in enumerate(data, start=1):
        # 필수 필드 추출
        professor_name = item.get("professor_name")
        university = item.get("university")
        department = item.get("department")
        title = item.get("title")

        # 스킵 조건: title 없음/공백
        if not title or not str(title).strip():
            print(f"[아이템 {idx}] title 누락 → 스킵")
            skipped_count += 1
            continue

        # lab_id 조회
        lab_id = _find_lab_id(db, professor_name, university, department, idx)
        if lab_id is None:
            skipped_count += 1
            continue

        # 선택적 필드 추출
        abstract = item.get("abstract")
        publication_year = item.get("publication_year")
        google_scholar_url = item.get("paper_url")
        citation_count = item.get("citation_count", 0)
        keywords = item.get("keywords")  # 있을 경우 JSON으로 그대로 저장 가능
        doi = item.get("doi")

        try:
            # summary 생성 & 저장은 이 함수 내부에서 자동 처리됨
            _ = insert_paper_with_summary(
                db=db,
                lab_id=lab_id,
                title=str(title).strip(),
                abstract_text=abstract,  # Paper.abstract 에 저장됨
                publication_year=publication_year,
                keywords=keywords,
                citation_count=citation_count if isinstance(citation_count, int) else 0,
                doi=doi,
                google_scholar_url=google_scholar_url,
            )
            inserted_count += 1
            print(f"✅ [아이템 {idx}] '{title}' 저장 완료 (lab_id={lab_id})")
        except Exception as e:
            # insert_paper_with_summary 내부에서 커밋 수행 → 실패 시에도 세션은 재사용 가능하도록 롤백
            db.rollback()
            print(f"[아이템 {idx}] DB 저장 오류: {e}")
            skipped_count += 1

    print("\n=== 완료 ===")
    print(f"✅ 삽입 완료: {inserted_count}개")
    print(f"⚠️  건너뜀: {skipped_count}개")
    return {"inserted": inserted_count, "skipped": skipped_count}
