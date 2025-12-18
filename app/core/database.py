# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import secrets

DATABASE_URL = secrets["database"]["url"]

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI용 DB 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
