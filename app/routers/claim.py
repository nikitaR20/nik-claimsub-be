from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from app import models, schemas, database
from app.database import get_db
router = APIRouter(prefix="/claims", tags=["Claims"])

'''
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()
'''

@router.get("/", response_model=List[schemas.ClaimOut])
def get_all_claims(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    db.expire_all()
    claims = (
        db.query(models.Claim)
        .order_by(models.Claim.created_at.desc())  # 🟢 ORDER BY latest
        .offset(skip)
        .limit(limit)
        .all()
    )
    return claims

@router.get("/{claim_id}", response_model=schemas.ClaimOut)
def get_claim(claim_id: UUID, db: Session = Depends(get_db)):
    """
    Retrieve a single claim by ID.
    """
    claim = db.query(models.Claim).filter(models.Claim.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim


@router.post("/", response_model=schemas.ClaimOut)
def create_claim(claim: schemas.ClaimCreate, db: Session = Depends(get_db)):
    """
    Create a new claim and return the created claim with ID.
    """
    new_claim = models.Claim(**claim.dict())
    db.add(new_claim)
    db.commit()
    db.refresh(new_claim)
    return new_claim
