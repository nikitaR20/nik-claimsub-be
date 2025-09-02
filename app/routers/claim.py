from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/claims", tags=["Claims"])

@router.get("/", response_model=List[schemas.ClaimOut])
def get_all_claims(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    claims = db.query(models.Claim).order_by(models.Claim.claim_date.desc()).offset(skip).limit(limit).all()
    return claims

@router.get("/{claim_id}", response_model=schemas.ClaimOut)
def get_claim(claim_id: UUID, db: Session = Depends(get_db)):
    claim = db.query(models.Claim).filter(models.Claim.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim

@router.post("/", response_model=schemas.ClaimOut)
def create_claim(claim: schemas.ClaimOut, db: Session = Depends(get_db)):
    new_claim = models.Claim(**claim.dict())
    db.add(new_claim)
    db.commit()
    db.refresh(new_claim)
    return new_claim
