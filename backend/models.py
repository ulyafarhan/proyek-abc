from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, required=True)
    hashed_password = Column(String, required=True)
    
    analyses = relationship("Analysis", back_populates="owner")
    journal_entries = relationship("JournalEntry", back_populates="owner")

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    analysis_type = Column(String) # "community" atau "behavior"
    result_json = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="analyses")

class JournalEntry(Base):
    __tablename__ = "journal_entries"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    content = Column(String)
    analysis_json = Column(String) # Hasil analisis Gemini untuk jurnal ini
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="journal_entries")