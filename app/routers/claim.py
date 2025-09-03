from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from typing import List
from uuid import UUID
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/claims", tags=["Claims"])


# -------------------- GET ALL CLAIMS --------------------
@router.get("/", response_model=List[schemas.ClaimOut])
def get_all_claims(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    """
    Fetch all claims with patient and provider info via relationships.
    """
    claims = (
        db.query(models.Claim)
        .options(
            joinedload(models.Claim.patient),   # fetch patient details
            joinedload(models.Claim.provider)   # fetch provider details
        )
        .order_by(models.Claim.claim_date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return claims


# -------------------- GET SINGLE CLAIM --------------------
@router.get("/{claim_id}", response_model=schemas.ClaimOut)
def get_claim(claim_id: UUID, db: Session = Depends(get_db)):
    """
    Fetch a single claim by ID with patient and provider details.
    """
    claim = (
        db.query(models.Claim)
        .options(
            joinedload(models.Claim.patient),
            joinedload(models.Claim.provider)
        )
        .filter(models.Claim.claim_id == claim_id)
        .first()
    )
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim


# -------------------- CREATE CLAIM --------------------
@router.post("/", response_model=schemas.ClaimOut)
def create_claim(claim: schemas.ClaimOut, db: Session = Depends(get_db)):
    """
    Create a new claim. Only store IDs in the Claim table; patient/provider info is fetched dynamically.
    """
    # Ensure the patient exists
    patient = db.query(models.Patient).filter(models.Patient.patient_id == claim.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Ensure the provider exists
    provider = db.query(models.Provider).filter(models.Provider.provider_id == claim.provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    # Only pass the actual claim fields (exclude patient/provider info)
    claim_data = claim.dict(exclude={"patient_age", "patient_gender", "patient_income",
                                     "patient_marital_status", "patient_employment_status",
                                     "provider_specialty", "provider_location"})

    new_claim = models.Claim(**claim_data)
    db.add(new_claim)
    db.commit()
    db.refresh(new_claim)
    return new_claim
