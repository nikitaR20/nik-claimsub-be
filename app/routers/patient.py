from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from uuid import UUID
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/patients", tags=["Patients"])

# -------------------- GET ALL PATIENTS --------------------
@router.get("/", response_model=List[schemas.PatientOut])
def get_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(6000, ge=1, le=10000),
    search: Optional[str] = Query(None, description="Search by name"),
    db: Session = Depends(get_db)
):
    """Get all patients with optional search and pagination"""
    query = db.query(models.Patient)
    
    # Apply search filter
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (models.Patient.first_name.ilike(search_pattern)) |
            (models.Patient.last_name.ilike(search_pattern))
        )
    
    patients = query.offset(skip).limit(limit).all()
    return patients

# -------------------- GET PATIENT BY ID --------------------
@router.get("/{patient_id}", response_model=schemas.PatientOut)
def get_patient(patient_id: UUID, db: Session = Depends(get_db)):
    """Get a specific patient by ID"""
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

# -------------------- CREATE PATIENT --------------------
@router.post("/", response_model=schemas.PatientOut)
def create_patient(patient_data: schemas.PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient"""
    new_patient = models.Patient(**patient_data.dict())
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient

# -------------------- UPDATE PATIENT --------------------
@router.put("/{patient_id}", response_model=schemas.PatientOut)
def update_patient(
    patient_id: UUID,
    patient_update: schemas.PatientUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing patient"""
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Update patient fields
    update_data = patient_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(patient, field, value)

    db.commit()
    db.refresh(patient)
    return patient

# -------------------- DELETE PATIENT --------------------
@router.delete("/{patient_id}")
def delete_patient(patient_id: UUID, db: Session = Depends(get_db)):
    """Delete a patient (only if no associated claims exist)"""
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Check if patient has any claims
    claims_count = db.query(models.Claim).filter(models.Claim.patient_id == patient_id).count()
    if claims_count > 0:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot delete patient with {claims_count} associated claims"
        )

    db.delete(patient)
    db.commit()
    return {"message": "Patient deleted successfully"}

# -------------------- PATIENT CLAIMS --------------------
@router.get("/{patient_id}/claims", response_model=List[schemas.ClaimOut])
def get_patient_claims(
    patient_id: UUID,
    include_related: bool = Query(True, description="Include related data"),
    db: Session = Depends(get_db)
):
    """Get all claims for a specific patient"""
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    query = db.query(models.Claim).filter(models.Claim.patient_id == patient_id)
    
    if include_related:
        query = query.options(
            joinedload(models.Claim.provider),
            joinedload(models.Claim.service_lines),
            joinedload(models.Claim.diagnoses)
        )
    
    claims = query.order_by(models.Claim.claim_date.desc()).all()
    return claims

# -------------------- PATIENT INSURANCES --------------------
@router.get("/{patient_id}/insurances", response_model=List[schemas.PatientInsuranceOut])
def get_patient_insurances(patient_id: UUID, db: Session = Depends(get_db)):
    """Get all insurance records for a specific patient"""
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    insurances = db.query(models.PatientInsurance).filter(
        models.PatientInsurance.patient_id == patient_id
    ).all()
    return insurances

@router.post("/{patient_id}/insurances", response_model=schemas.PatientInsuranceOut)
def add_patient_insurance(
    patient_id: UUID,
    insurance_data: schemas.PatientInsuranceCreate,
    db: Session = Depends(get_db)
):
    """Add insurance information for a patient"""
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Ensure patient_id matches
    insurance_dict = insurance_data.dict()
    insurance_dict["patient_id"] = patient_id

    new_insurance = models.PatientInsurance(**insurance_dict)
    db.add(new_insurance)
    db.commit()
    db.refresh(new_insurance)
    return new_insurance

@router.put("/{patient_id}/insurances/{insurance_id}", response_model=schemas.PatientInsuranceOut)
def update_patient_insurance(
    patient_id: UUID,
    insurance_id: UUID,
    insurance_update: schemas.PatientInsuranceCreate,
    db: Session = Depends(get_db)
):
    """Update patient insurance information"""
    insurance = db.query(models.PatientInsurance).filter(
        models.PatientInsurance.insurance_id == insurance_id,
        models.PatientInsurance.patient_id == patient_id
    ).first()
    
    if not insurance:
        raise HTTPException(status_code=404, detail="Insurance record not found")

    # Update insurance fields
    update_data = insurance_update.dict()
    for field, value in update_data.items():
        if field != "patient_id":  # Don't update patient_id
            setattr(insurance, field, value)

    db.commit()
    db.refresh(insurance)
    return insurance

@router.delete("/{patient_id}/insurances/{insurance_id}")
def delete_patient_insurance(
    patient_id: UUID,
    insurance_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete patient insurance record"""
    insurance = db.query(models.PatientInsurance).filter(
        models.PatientInsurance.insurance_id == insurance_id,
        models.PatientInsurance.patient_id == patient_id
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

# -------------------- PATIENT SEARCH --------------------
@router.get("/search/advanced")
def search_patients_advanced(
    first_name: Optional[str] = Query(None),
    last_name: Optional[str] = Query(None),
    birth_date: Optional[str] = Query(None),
    phone: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Advanced patient search with multiple criteria"""
    query = db.query(models.Patient)
    
    if first_name:
        query = query.filter(models.Patient.first_name.ilike(f"%{first_name}%"))
    if last_name:
        query = query.filter(models.Patient.last_name.ilike(f"%{last_name}%"))
    if birth_date:
        from datetime import datetime
        try:
            birth_date_parsed = datetime.strptime(birth_date, "%Y-%m-%d").date()
            query = query.filter(models.Patient.birth_date == birth_date_parsed)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid birth_date format. Use YYYY-MM-DD")
    if phone:
        query = query.filter(models.Patient.phone.ilike(f"%{phone}%"))
    if email:
        query = query.filter(models.Patient.email.ilike(f"%{email}%"))
    
    patients = query.limit(50).all()  # Limit results to prevent large responses
    return patients

# -------------------- LEGACY COMPATIBILITY --------------------
@router.post("/legacy", response_model=schemas.PatientOut)
def create_patient_legacy(patient: schemas.PatientCreate_Legacy, db: Session = Depends(get_db)):
    """Legacy endpoint for backward compatibility with existing frontend"""
    # Convert legacy fields to new format
    patient_data = {
        "first_name": patient.first_name or "",
        "last_name": patient.last_name or "",
        "birth_date": "1900-01-01",  # Default date if not provided
        "gender": patient.patient_gender or "X",
        "age": patient.patient_age,
        "income": patient.patient_income,
        "marital_status": patient.patient_marital_status,
        "employment_status": patient.patient_employment_status
    }
    
    # Remove None values
    patient_data = {k: v for k, v in patient_data.items() if v is not None}
    
    new_patient = models.Patient(**patient_data)
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient