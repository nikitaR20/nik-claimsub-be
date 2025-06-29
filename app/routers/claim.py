from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from app import models, schemas, database

router = APIRouter(prefix="/claims", tags=["Claims"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=List[schemas.ClaimOut])
def get_all_claims(db: Session = Depends(get_db)):
    return db.query(models.Claim).all()

@router.get("/{claim_id}", response_model=schemas.ClaimOut)
def get_claim(claim_id: UUID, db: Session = Depends(get_db)):
    claim = db.query(models.Claim).filter(models.Claim.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim

@router.post("/", response_model=schemas.ClaimOut)
def create_claim(claim: schemas.ClaimCreate, db: Session = Depends(get_db)):
    new_claim = models.Claim(**claim.dict())
    db.add(new_claim)
    db.commit()
    db.refresh(new_claim)
    return new_claim
