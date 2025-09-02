from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/providers", tags=["Providers"])

@router.get("/", response_model=List[schemas.ProviderOut])
def get_providers(db: Session = Depends(get_db)):
    return db.query(models.Provider).all()
