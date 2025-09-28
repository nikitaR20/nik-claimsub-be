from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from uuid import UUID
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/providers", tags=["Providers"])

# -------------------- GET ALL PROVIDERS --------------------
@router.get("/", response_model=List[schemas.ProviderOut])
def get_providers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None, description="Search by name or NPI"),
    specialty: Optional[str] = Query(None, description="Filter by specialty"),
    db: Session = Depends(get_db)
):
    """Get all providers with optional search and filtering"""
    query = db.query(models.Provider)
    
    # Apply search filter
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (models.Provider.first_name.ilike(search_pattern)) |
            (models.Provider.last_name.ilike(search_pattern)) |
            (models.Provider.npi.ilike(search_pattern))
        )
    
    # Apply specialty filter
    if specialty:
        query = query.filter(models.Provider.specialty.ilike(f"%{specialty}%"))
    
    providers = query.offset(skip).limit(limit).all()
    return providers

# -------------------- GET PROVIDER BY ID --------------------
@router.get("/{provider_id}", response_model=schemas.ProviderOut)
def get_provider(provider_id: UUID, db: Session = Depends(get_db)):
    """Get a specific provider by ID"""
    provider = db.query(models.Provider).filter(models.Provider.provider_id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider

# -------------------- GET PROVIDER BY NPI --------------------
@router.get("/npi/{npi}", response_model=schemas.ProviderOut)
def get_provider_by_npi(npi: str, db: Session = Depends(get_db)):
    """Get a provider by NPI number"""
    provider = db.query(models.Provider).filter(models.Provider.npi == npi).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider

# -------------------- CREATE PROVIDER --------------------
@router.post("/", response_model=schemas.ProviderOut)
def create_provider(provider_data: schemas.ProviderCreate, db: Session = Depends(get_db)):
    """Create a new provider"""
    # Check if NPI already exists
    existing_provider = db.query(models.Provider).filter(models.Provider.npi == provider_data.npi).first()
    if existing_provider:
        raise HTTPException(status_code=400, detail="Provider with this NPI already exists")
    
    new_provider = models.Provider(**provider_data.dict())
    db.add(new_provider)
    db.commit()
    db.refresh(new_provider)
    return new_provider

# -------------------- UPDATE PROVIDER --------------------
@router.put("/{provider_id}", response_model=schemas.ProviderOut)
def update_provider(
    provider_id: UUID,
    provider_update: schemas.ProviderUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing provider"""
    provider = db.query(models.Provider).filter(models.Provider.provider_id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    # Check if NPI is being updated and if it conflicts
    if provider_update.npi and provider_update.npi != provider.npi:
        existing_provider = db.query(models.Provider).filter(
            models.Provider.npi == provider_update.npi,
            models.Provider.provider_id != provider_id
        ).first()
        if existing_provider:
            raise HTTPException(status_code=400, detail="Provider with this NPI already exists")

    # Update provider fields
    update_data = provider_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(provider, field, value)

    db.commit()
    db.refresh(provider)
    return provider

# -------------------- DELETE PROVIDER --------------------
@router.delete("/{provider_id}")
def delete_provider(provider_id: UUID, db: Session = Depends(get_db)):
    """Delete a provider (only if no associated claims exist)"""
    provider = db.query(models.Provider).filter(models.Provider.provider_id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    # Check if provider has any claims
    claims_count = db.query(models.Claim).filter(models.Claim.provider_id == provider_id).count()
    if claims_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete provider with {claims_count} associated claims"
        )

    db.delete(provider)
    db.commit()
    return {"message": "Provider deleted successfully"}

# -------------------- PROVIDER CLAIMS --------------------
@router.get("/{provider_id}/claims", response_model=List[schemas.ClaimOut])
def get_provider_claims(
    provider_id: UUID,
    include_related: bool = Query(True, description="Include related data"),
    db: Session = Depends(get_db)
):
    """Get all claims for a specific provider"""
    provider = db.query(models.Provider).filter(models.Provider.provider_id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    query = db.query(models.Claim).filter(models.Claim.provider_id == provider_id)
    
    if include_related:
        query = query.options(
            joinedload(models.Claim.patient),
            joinedload(models.Claim.service_lines),
            joinedload(models.Claim.diagnoses)
        )
    
    claims = query.order_by(models.Claim.claim_date.desc()).all()
    return claims

# -------------------- PROVIDER STATISTICS --------------------
@router.get("/{provider_id}/statistics")
def get_provider_statistics(provider_id: UUID, db: Session = Depends(get_db)):
    """Get statistics for a specific provider"""
    provider = db.query(models.Provider).filter(models.Provider.provider_id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    from sqlalchemy import func
    
    # Basic statistics
    total_claims = db.query(func.count(models.Claim.claim_id)).filter(
        models.Claim.provider_id == provider_id
    ).scalar()
    
    total_billed = db.query(func.sum(models.Claim.total_charge)).filter(
        models.Claim.provider_id == provider_id
    ).scalar() or 0
    
    total_paid = db.query(func.sum(models.Claim.amount_paid)).filter(
        models.Claim.provider_id == provider_id
    ).scalar() or 0
    
    # Claims by status
    claims_by_status = db.query(
        models.Claim.claim_status,
        func.count(models.Claim.claim_id).label('count')
    ).filter(
        models.Claim.provider_id == provider_id
    ).group_by(models.Claim.claim_status).all()
    
    return {
        "provider_id": provider_id,
        "total_claims": total_claims,
        "total_billed": total_billed,
        "total_paid": total_paid,
        "collection_rate": (total_paid / total_billed * 100) if total_billed > 0 else 0,
        "claims_by_status": [{"status": status, "count": count} for status, count in claims_by_status]
    }

# -------------------- PROVIDER SPECIALTIES --------------------
@router.get("/specialties/list")
def get_provider_specialties(db: Session = Depends(get_db)):
    """Get list of all provider specialties"""
    from sqlalchemy import func
    
    specialties = db.query(
        models.Provider.specialty,
        func.count(models.Provider.provider_id).label('provider_count')
    ).filter(
        models.Provider.specialty.isnot(None)
    ).group_by(models.Provider.specialty).all()
    
    return [{"specialty": specialty, "provider_count": count} for specialty, count in specialties if specialty]

# -------------------- VALIDATE NPI --------------------
@router.get("/validate/npi/{npi}")
def validate_npi(npi: str, db: Session = Depends(get_db)):
    """Validate if NPI exists and get basic info"""
    provider = db.query(models.Provider).filter(models.Provider.npi == npi).first()
    
    if provider:
        return {
            "valid": True,
            "provider_id": provider.provider_id,
            "name": f"{provider.first_name} {provider.last_name}",
            "specialty": provider.specialty
        }
    else:
        return {"valid": False, "message": "NPI not found in system"}