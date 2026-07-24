from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./llm_eval.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_prompts = Column(Integer, default=0)
    models_tested = Column(JSON, default=list)
    results = relationship("EvaluationResult", back_populates="experiment")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    prompt = Column(Text, nullable=False)
    expected = Column(Text, nullable=True)
    model_name = Column(String, nullable=False)
    response = Column(Text)
    correctness = Column(Float, default=0.0)
    relevance = Column(Float, default=0.0)
    faithfulness = Column(Float, default=0.0)
    completeness = Column(Float, default=0.0)
    hallucination_free = Column(Float, default=0.0)
    overall_score = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    cached = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    experiment = relationship("Experiment", back_populates="results")


class CacheEntry(Base):
    __tablename__ = "cache_entries"
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String, unique=True, index=True)
    response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
