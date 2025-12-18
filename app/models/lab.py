# lab.py
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.sql import func
from app.core.database import Base

class Lab(Base):
    __tablename__ = "labs"

    lab_id = Column(Integer, primary_key=True, autoincrement=True)
    professor_name = Column(String(100), nullable=False)
    lab_name = Column(String(150))
    university = Column(String(100))
    department = Column(String(100))
    major = Column(String(100)) 
    summary = Column(Text)
    homepage_url = Column(String(255))
    google_scholar_url = Column(String(255))
    image_url = Column(String(255))
    education_text = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
