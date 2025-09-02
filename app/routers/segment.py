from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/segments", tags=["Segments"])

@router.get("/", response_model=List[schemas.SegmentOut])
def get_segments(db: Session = Depends(get_db)):
    return db.query(models.Segment).all()
