from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from uuid import UUID
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/insurances", tags=["Insurance"])

# -------------------- GET ALL INSURANCE RECORDS --------------------
@router.get("/", response_model=List[schemas.PatientInsuranceOut])
def get_all_insurances(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    insurance_type: Optional[str] = Query(None, description="Filter by insurance type"),
    patient_id: Optional[UUID] = Query(None, description="Filter by patient ID"),
    db: Session = Depends(get_db)
):
    """Get all insurance records with optional filtering"""
    query = db.query(models.PatientInsurance).options(joinedload(models.PatientInsurance.patient))
    
    # Apply filters
    if insurance_type:
        query = query.filter(models.PatientInsurance.insurance_type == insurance_type)
    if patient_id:
        query = query.filter(models.PatientInsurance.patient_id == patient_id)
    
    insurances = query.offset(skip).limit(limit).all()
    return insurances

# -------------------- GET INSURANCE BY ID --------------------
@router.get("/{insurance_id}", response_model=schemas.PatientInsuranceOut)
def get_insurance(insurance_id: UUID, db: Session = Depends(get_db)):
    """Get a specific insurance record by ID"""
    insurance = (
        db.query(models.PatientInsurance)
        .options(joinedload(models.PatientInsurance.patient))
        .filter(models.PatientInsurance.insurance_id == insurance_id)
        .first()
    )
    if not insurance:
        raise HTTPException(status_code=404, detail="Insurance record not found")
    return insurance

# -------------------- CREATE INSURANCE --------------------
@router.post("/", response_model=schemas.PatientInsuranceOut)
def create_insurance(insurance_data: schemas.PatientInsuranceCreate, db: Session = Depends(get_db)):
    """Create a new insurance record"""
    # Validate patient exists
    patient = db.query(models.Patient).filter(models.Patient.patient_id == insurance_data.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    new_insurance = models.PatientInsurance(**insurance_data.dict())
    db.add(new_insurance)
    db.commit()
    db.refresh(new_insurance)
    return new_insurance

# -------------------- UPDATE INSURANCE --------------------
@router.put("/{insurance_id}", response_model=schemas.PatientInsuranceOut)
def update_insurance(
    insurance_id: UUID,
    insurance_update: schemas.PatientInsuranceCreate,
    db: Session = Depends(get_db)
):
    """Update an existing insurance record"""
    insurance = db.query(models.PatientInsurance).filter(
        models.PatientInsurance.insurance_id == insurance_id
    ).first()
    
    if not insurance:
        raise HTTPException(status_code=404, detail="Insurance record not found")

    # Update insurance fields
    update_data = insurance_update.dict()
    for field, value in update_data.items():
        setattr(insurance, field, value)

    db.commit()
    db.refresh(insurance)
    return insurance

# -------------------- DELETE INSURANCE --------------------
@router.delete("/{insurance_id}")
def delete_insurance(insurance_id: UUID, db: Session = Depends(get_db)):
    """Delete an insurance record"""
    insurance = db.query(models.PatientInsurance).filter(
        models.PatientInsurance.insurance_id == insurance_id
    ).first()
    
    if not insurance:
        raise HTTPException(status_code=404, detail="Insurance record not found")

    # Check if insurance is used in any claims
    claims_count = db.query(models.Claim).filter(models.Claim.insurance_id == insurance_id).count()
    if claims_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete insurance record used in {claims_count} claims"
        )

    db.delete(insurance)
    db.commit()
    return {"message": "Insurance record deleted successfully"}

# -------------------- INSURANCE VERIFICATION --------------------
@router.post("/{insurance_id}/verify")
def verify_insurance(insurance_id: UUID, db: Session = Depends(get_db)):
    """Verify insurance eligibility (placeholder for real implementation)"""
    insurance = db.query(models.PatientInsurance).filter(
        models.PatientInsurance.insurance_id == insurance_id
    ).first()
    
    if not insurance:
        raise HTTPException(status_code=404, detail="Insurance record not found")
    
    # This would integrate with real insurance verification APIs
    # For now, return mock verification result
    return {
        "insurance_id": insurance_id,
        "verified": True,
        "effective_date": insurance.effective_date,
        "termination_date": insurance.termination_date,
        "coverage_status": "Active",
        "copay_amount": 25.00,
        "deductible_remaining": 500.00,
        "verification_date": "2024-01-01T00:00:00Z",
        "message": "Coverage verified successfully"
    }

# -------------------- INSURANCE TYPES --------------------
@router.get("/types/summary")
def get_insurance_types_summary(db: Session = Depends(get_db)):
    """Get summary of insurance types in the system"""
    from sqlalchemy import func
    
    types_summary = db.query(
        models.PatientInsurance.insurance_type,
        func.count(models.PatientInsurance.insurance_id).label('count')
    ).group_by(models.PatientInsurance.insurance_type).all()
    
    return [{"insurance_type": type_name, "count": count} for type_name, count in types_summary]

# -------------------- CLAIMS BY INSURANCE --------------------
@router.get("/{insurance_id}/claims", response_model=List[schemas.ClaimOut])
def get_claims_by_insurance(
    insurance_id: UUID,
    include_related: bool = Query(True, description="Include related data"),
    db: Session = Depends(get_db)
):
    """Get all claims for a specific insurance"""
    insurance = db.query(models.PatientInsurance).filter(
        models.PatientInsurance.insurance_id == insurance_id
    ).first()
    if not insurance:
        raise HTTPException(status_code=404, detail="Insurance record not found")

    query = db.query(models.Claim).filter(models.Claim.insurance_id == insurance_id)
    
    if include_related:
        query = query.options(
            joinedload(models.Claim.patient),
            joinedload(models.Claim.provider),
            joinedload(models.Claim.service_lines),
            joinedload(models.Claim.diagnoses)
        )
    
    claims = query.order_by(models.Claim.claim_date.desc()).all()
    return claims

# -------------------- INSURANCE VALIDATION --------------------
@router.post("/validate/policy")
def validate_policy_number(
    insurance_type: str,
    policy_number: str,
    db: Session = Depends(get_db)
):
    """Validate if a policy number exists for given insurance type"""
    existing_policy = db.query(models.PatientInsurance).filter(
        models.PatientInsurance.insurance_type == insurance_type,
        models.PatientInsurance.policy_number == policy_number
    ).first()
    
    if existing_policy:
        return {
            "valid": True,
            "insurance_id": existing_policy.insurance_id,
            "patient_id": existing_policy.patient_id,
            "plan_name": existing_policy.plan_name,
            "effective_date": existing_policy.effective_date
        }
    else:
        return {"valid": False, "message": "Policy number not found"}

# -------------------- INSURANCE STATISTICS --------------------
@router.get("/{insurance_id}/statistics")
def get_insurance_statistics(insurance_id: UUID, db: Session = Depends(get_db)):
    """Get statistics for a specific insurance record"""
    insurance = db.query(models.PatientInsurance).filter(
        models.PatientInsurance.insurance_id == insurance_id
    ).first()
    if not insurance:
        raise HTTPException(status_code=404, detail="Insurance record not found")

    from sqlalchemy import func
    
    # Basic statistics
    total_claims = db.query(func.count(models.Claim.claim_id)).filter(
        models.Claim.insurance_id == insurance_id
    ).scalar()
    
    total_billed = db.query(func.sum(models.Claim.total_charge)).filter(
        models.Claim.insurance_id == insurance_id
    ).scalar() or 0
    
    total_paid = db.query(func.sum(models.Claim.amount_paid)).filter(
        models.Claim.insurance_id == insurance_id
    ).scalar() or 0
    
    # Claims by status
    claims_by_status = db.query(
        models.Claim.claim_status,
        func.count(models.Claim.claim_id).label('count')
    ).filter(
        models.Claim.insurance_id == insurance_id
    ).group_by(models.Claim.claim_status).all()
    
    return {
        "insurance_id": insurance_id,
        "insurance_type": insurance.insurance_type,
        "plan_name": insurance.plan_name,
        "total_claims": total_claims,
        "total_billed": total_billed,
        "total_paid": total_paid,
        "claims_by_status": [{"status": status, "count": count} for status, count in claims_by_status]
    }