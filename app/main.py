from fastapi import FastAPI
from app.core.database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Laser Backend API")

@app.get("/")
def root():
    return {"message": "Welcome to Laser Backend!"}


# 임시 api 호출
# from fastapi import FastAPI, Depends
# from sqlalchemy.orm import Session
# from app.core.database import Base, engine, get_db
# from app.services.db_service import insert_lab_with_summary, insert_paper_with_summary

# Base.metadata.create_all(bind=engine)
# app = FastAPI(title="Laser Backend API")

# @app.get("/")
# def root():
#     return {"message": "Laser Backend Running 🚀"}

# @app.post("/labs/summary-test")
# def create_lab_with_llm(db: Session = Depends(get_db)):
#     homepage_text = """
#     본 연구실은 인공지능, 컴퓨터 비전 및 로봇 학습을 중점적으로 연구합니다.
#     딥러닝 기반 영상 인식, 객체 탐지, 자율주행 시스템을 개발하며 다양한 산업 응용을 목표로 합니다.
#     """
#     lab = insert_lab_with_summary(
#         db,
#         professor_name="홍길동",
#         lab_name="AI Vision Lab",
#         university="KAIST",
#         department="전산학부",
#         homepage_text=homepage_text,
#         homepage_url="https://example.com",
#     )
#     return lab
