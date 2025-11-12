# app/services/lab_loader_service.py
import json
from pathlib import Path
from sqlalchemy.orm import Session
from app.models.lab import Lab
from app.services.llm_service import generate_lab_summary


def load_lab_json_to_mysql(db: Session, file_path: str):
    """
    연구실 JSONL 파일을 읽어서 MySQL에 삽입하는 함수.
    summary는 generate_lab_summary()를 사용해 생성 후 저장한다.
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"❌ 연구실 데이터 파일을 찾을 수 없습니다: {file_path}")

    inserted_count = 0
    skipped_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                lab_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[라인 {line_no}] JSON 파싱 오류: {e}")
                skipped_count += 1
                continue

            # 필드 추출
            professor_name = lab_data.get("professor_name")
            lab_name = lab_data.get("lab_name")  # nullable
            university = lab_data.get("university")
            department = lab_data.get("department")
            homepage_url = lab_data.get("homepage_url")
            google_scholar_url = lab_data.get("google_scholar_url")
            image_url = lab_data.get("image_url")
            education_text = lab_data.get("education_text")

            # 필수 필드 체크
            before_summary = lab_data.get("before_summary")
            if not professor_name or not before_summary:
                print(f"[라인 {line_no}] 필수 데이터 누락 (교수명 or before_summary)")
                skipped_count += 1
                continue

            # 연구실 소개 텍스트 추출
            lab_intro_text = None
            for section in before_summary:
                if section.get("type") == "research_area":
                    lab_intro_text = section.get("content")
                    break

            if not lab_intro_text or len(lab_intro_text.strip()) == 0:
                print(f"[라인 {line_no}] 연구실 소개 내용 비어 있음 → 건너뜀")
                skipped_count += 1
                continue

            # LLM 요약 생성
            summary = generate_lab_summary(lab_intro_text, department)

            # DB 저장
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

            try:
                db.add(lab)
                db.commit()
                db.refresh(lab)
                inserted_count += 1
                print(f"✅ [{line_no}] {professor_name} 연구실 저장 완료 (ID={lab.lab_id})")
            except Exception as e:
                db.rollback()
                print(f"[라인 {line_no}] DB 저장 오류: {e}")
                skipped_count += 1

    print(f"\n=== 완료 ===")
    print(f"✅ 삽입 완료: {inserted_count}개")
    print(f"⚠️  건너뜀: {skipped_count}개")
    return {"inserted": inserted_count, "skipped": skipped_count}
