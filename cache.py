import hashlib
from sqlalchemy.orm import Session
from database import CacheEntry
from datetime import datetime


def _make_key(prompt: str, model_name: str) -> str:
    return hashlib.sha256(f"{prompt}||{model_name}".encode()).hexdigest()


def get_cached(db: Session, prompt: str, model_name: str):
    key = _make_key(prompt, model_name)
    entry = db.query(CacheEntry).filter(CacheEntry.cache_key == key).first()
    return entry.response if entry else None


def set_cache(db: Session, prompt: str, model_name: str, response: str):
    key = _make_key(prompt, model_name)
    entry = CacheEntry(cache_key=key, response=response, created_at=datetime.utcnow())
    db.add(entry)
    db.commit()
