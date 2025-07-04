from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app import models, database, schemas

router = APIRouter(prefix="/segments", tags=["Segments"])  # Changed path to /segments


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_model=List[schemas.SegmentOut])
def get_segments(db: Session = Depends(get_db)):
    """
    Retrieve list of all segments.
    """
    return db.query(models.Segment).all()
