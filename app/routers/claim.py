from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from typing import List
from uuid import UUID
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/claims", tags=["Claims"])

# -------------------- GET ALL CLAIMS --------------------
@router.get("/", response_model=List[schemas.ClaimOut])
def get_all_claims(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    claims = (
        db.query(models.Claim)
        .options(joinedload(models.Claim.patient), joinedload(models.Claim.provider))
        .order_by(models.Claim.claim_date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return claims

# -------------------- CREATE OR UPDATE CLAIM --------------------
@router.post("/", response_model=schemas.ClaimOut)
def create_or_update_claim(claim: schemas.ClaimCreate, db: Session = Depends(get_db)):
    # Check if patient exists
    patient = db.query(models.Patient).filter(models.Patient.patient_id == claim.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Check if provider exists
    provider = db.query(models.Provider).filter(models.Provider.provider_id == claim.provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    # If claim_id exists -> update
    if getattr(claim, "claim_id", None):
        existing_claim = db.query(models.Claim).filter(models.Claim.claim_id == claim.claim_id).first()
        if not existing_claim:
            raise HTTPException(status_code=404, detail="Claim not found")
        
        # Update all fields
        for field, value in claim.dict(exclude_unset=True).items():
            setattr(existing_claim, field, value)
        
        db.commit()
        db.refresh(existing_claim)
        updated_claim = existing_claim
    else:
        # Create new claim
        new_claim = models.Claim(**claim.dict())
        db.add(new_claim)
        db.commit()
        db.refresh(new_claim)
        updated_claim = new_claim

    # Return claim with nested patient and provider
    return (
        db.query(models.Claim)
        .options(joinedload(models.Claim.patient), joinedload(models.Claim.provider))
        .filter(models.Claim.claim_id == updated_claim.claim_id)
        .first()
    )
