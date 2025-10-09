from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Paper(Base):
    __tablename__ = "papers"

    paper_id = Column(Integer, primary_key=True, autoincrement=True)
    lab_id = Column(Integer, ForeignKey("labs.lab_id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    summary = Column(Text)
    publication_year = Column(Integer)
    keywords = Column(JSON)
    citation_count = Column(Integer, default=0)
    doi = Column(String(100))
    google_scholar_url = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now())

    lab = relationship("Lab", backref="papers")
