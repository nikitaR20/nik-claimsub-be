from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app import models, database, schemas
from app.database import get_db
router = APIRouter(prefix="/providers", tags=["Providers"])

'''
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()
'''

@router.get("/", response_model=List[schemas.ProviderOut])
def get_providers(db: Session = Depends(get_db)):
    """
    Retrieve list of all providers.
    """
    return db.query(models.Provider).all()
