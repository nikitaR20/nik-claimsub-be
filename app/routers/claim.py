from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from uuid import UUID
from app import models, schemas
from app.database import get_db
from app.ai.semantic_search import enhanced_semantic_search

router = APIRouter(prefix="/claims", tags=["Claims"])

# -------------------- CREATE CLAIM WITH AI --------------------
@router.post("/with-ai-suggestions", response_model=schemas.ClaimOut)
def create_claim_with_ai_suggestions(
    claim_data: schemas.ClaimCreate, 
    db: Session = Depends(get_db)
):
    """Create a new claim with AI-powered code suggestions for service lines and diagnoses"""
    # Validate patient and provider first
    patient = db.query(models.Patient).filter(models.Patient.patient_id == claim_data.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    provider = db.query(models.Provider).filter(models.Provider.provider_id == claim_data.provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    if claim_data.insurance_id:
        insurance = db.query(models.PatientInsurance).filter(
            models.PatientInsurance.insurance_id == claim_data.insurance_id
        ).first()
        if not insurance:
            raise HTTPException(status_code=404, detail="Insurance not found")

    # Build context for AI suggestions
    context_parts = []
    if claim_data.coverage_notes:
        context_parts.append(claim_data.coverage_notes)
    if claim_data.additional_claim_info:
        context_parts.append(claim_data.additional_claim_info)
    
    analysis_context = " ".join(context_parts)

    # Get AI suggestions if context available
    ai_suggestions = None
    if analysis_context.strip():
        # Use enhanced search that returns ICD, CPT, and HCPCS separately
        icd_results, cpt_results, hcpcs_results = enhanced_semantic_search(analysis_context, top_n=5)
        
        ai_suggestions = {
            "suggested_diagnoses": icd_results,
            "suggested_cpt_procedures": cpt_results,
            "suggested_hcpcs_items": hcpcs_results,
            "analysis_context": analysis_context
        }
        
        # Auto-add diagnosis codes if none provided
        if not claim_data.diagnoses and icd_results:
            for i, icd_result in enumerate(icd_results[:4]):  # Max 4 diagnoses
                claim_data.diagnoses.append(schemas.ClaimDiagnosisCreate(
                    position=i + 1,
                    icd10_code=icd_result['code'],
                    diagnosis_pointer=chr(65 + i)  # A, B, C, D
                ))
        
        # Auto-add service lines if none provided
        if not claim_data.service_lines:
            # Determine best procedure match between CPT and HCPCS
            best_procedure = None
            best_code_type = "CPT"
            
            if cpt_results and hcpcs_results:
                best_cpt = cpt_results[0] if cpt_results else None
                best_hcpcs = hcpcs_results[0] if hcpcs_results else None
                
                if best_cpt and best_hcpcs:
                    if best_cpt.get("similarity", 0) >= best_hcpcs.get("similarity", 0):
                        best_procedure = best_cpt
                        best_code_type = "CPT"
                    else:
                        best_procedure = best_hcpcs
                        best_code_type = "HCPCS"
                elif best_cpt:
                    best_procedure = best_cpt
                    best_code_type = "CPT"
                elif best_hcpcs:
                    best_procedure = best_hcpcs
                    best_code_type = "HCPCS"
            elif cpt_results:
                best_procedure = cpt_results[0]
                best_code_type = "CPT"
            elif hcpcs_results:
                best_procedure = hcpcs_results[0]
                best_code_type = "HCPCS"
            
            # Create service line with best match
            if best_procedure:
                claim_data.service_lines.append(schemas.ClaimServiceLineCreate(
                    service_date_from=claim_data.claim_date,
                    place_of_service_code="11",
                    cpt_hcpcs_code=best_procedure['code'],
                    code_type=best_code_type,
                    charge_amount=100.00  # Default charge amount
                ))

    # Create the claim using the standard creation process
    claim = create_claim(claim_data, db)
    
    # Add AI suggestions to the response
    claim_dict = claim.__dict__.copy()
    claim_dict['ai_suggestions'] = ai_suggestions
    
    return claim

# -------------------- GET ALL CLAIMS --------------------
@router.get("/", response_model=schemas.ClaimListResponse)
def get_all_claims(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None, description="Filter by claim status"),
    patient_id: Optional[UUID] = Query(None, description="Filter by patient ID"),
    provider_id: Optional[UUID] = Query(None, description="Filter by provider ID"),
    include_related: bool = Query(True, description="Include related patient/provider data"),
    db: Session = Depends(get_db)
):
    """Get all claims with optional filtering and pagination"""
    query = db.query(models.Claim)
    
    # Apply filters
    if status:
        query = query.filter(models.Claim.claim_status == status)
    if patient_id:
        query = query.filter(models.Claim.patient_id == patient_id)
    if provider_id:
        query = query.filter(models.Claim.provider_id == provider_id)
    
    # Include related data if requested
    if include_related:
        query = query.options(
            joinedload(models.Claim.patient),
            joinedload(models.Claim.provider),
            joinedload(models.Claim.insurance),
            joinedload(models.Claim.service_lines),
            joinedload(models.Claim.diagnoses)
        )
    
    # Get total count for pagination
    total_count = query.count()
    
    # Apply pagination and ordering
    claims = (
        query.order_by(models.Claim.claim_date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    return schemas.ClaimListResponse(
        claims=claims,
        total_count=total_count,
        page=(skip // limit) + 1,
        page_size=limit
    )

# -------------------- GET CLAIM BY ID --------------------
@router.get("/{claim_id}", response_model=schemas.ClaimOut)
def get_claim(claim_id: UUID, db: Session = Depends(get_db)):
    """Get a specific claim by ID with all related data"""
    claim = (
        db.query(models.Claim)
        .options(
            joinedload(models.Claim.patient),
            joinedload(models.Claim.provider),
            joinedload(models.Claim.insurance),
            joinedload(models.Claim.service_lines),
            joinedload(models.Claim.diagnoses),
            joinedload(models.Claim.documents)
        )
        .filter(models.Claim.claim_id == claim_id)
        .first()
    )
    
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    
    return claim

# -------------------- CREATE CLAIM --------------------
@router.post("/", response_model=schemas.ClaimOut)
def create_claim(claim_data: schemas.ClaimCreate, db: Session = Depends(get_db)):
    """Create a new claim with service lines and diagnoses"""
    # Validate patient exists
    patient = db.query(models.Patient).filter(models.Patient.patient_id == claim_data.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Validate provider exists
    provider = db.query(models.Provider).filter(models.Provider.provider_id == claim_data.provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    # Validate insurance if provided
    if claim_data.insurance_id:
        insurance = db.query(models.PatientInsurance).filter(
            models.PatientInsurance.insurance_id == claim_data.insurance_id
        ).first()
        if not insurance:
            raise HTTPException(status_code=404, detail="Insurance not found")

    # Create main claim record
    claim_dict = claim_data.dict(exclude={"service_lines", "diagnoses"})
    new_claim = models.Claim(**claim_dict)
    db.add(new_claim)
    db.flush()  # Get the claim_id without committing

    # Create service lines with code type support
    for line_data in claim_data.service_lines:
        service_line_dict = line_data.dict()
        
        # Ensure code_type is set
        if 'code_type' not in service_line_dict or not service_line_dict['code_type']:
            # Try to determine code type based on the code
            code = service_line_dict.get('cpt_hcpcs_code', '')
            if code and len(code) >= 3:
                if code[0].isalpha():
                    service_line_dict['code_type'] = 'HCPCS'
                else:
                    service_line_dict['code_type'] = 'CPT'
            else:
                service_line_dict['code_type'] = 'CPT'  # Default
        
        service_line = models.ClaimServiceLine(
            claim_id=new_claim.claim_id,
            **service_line_dict
        )
        db.add(service_line)

    # Create diagnoses
    for diag_data in claim_data.diagnoses:
        diagnosis = models.ClaimDiagnosis(
            claim_id=new_claim.claim_id,
            **diag_data.dict()
        )
        db.add(diagnosis)

    db.commit()
    db.refresh(new_claim)

    # Return claim with all related data
    return get_claim(new_claim.claim_id, db)

# -------------------- AI CODE SUGGESTIONS FOR EXISTING CLAIMS --------------------
@router.post("/{claim_id}/ai-suggestions", response_model=schemas.EnhancedCodeSuggestionResponse)
def get_ai_suggestions_for_claim(
    claim_id: UUID,
    notes: str = Query(..., description="Clinical notes or context for suggestions"),
    top_n: int = Query(3, ge=1, le=10),
    db: Session = Depends(get_db)
):
    """Get AI suggestions for an existing claim"""
    claim = db.query(models.Claim).filter(models.Claim.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    
    # Combine claim context with provided notes
    context_parts = [notes]
    if claim.coverage_notes:
        context_parts.append(claim.coverage_notes)
    if claim.additional_claim_info:
        context_parts.append(claim.additional_claim_info)
    
    analysis_context = " ".join(context_parts)
    
    # Get AI suggestions
    icd_results, cpt_results, hcpcs_results = enhanced_semantic_search(analysis_context, top_n=top_n)
    
    icd_suggestions = [
        schemas.CodeSuggestion(
            code=r["code"],
            description=r["description"],
            confidence=r.get("similarity", 0.5),
            code_type="ICD10"
        )
        for r in icd_results
    ]
    
    cpt_suggestions = [
        schemas.CodeSuggestion(
            code=r["code"],
            description=r["description"],
            confidence=r.get("similarity", 0.5),
            code_type="CPT"
        )
        for r in cpt_results
    ]
    
    hcpcs_suggestions = [
        schemas.HCPCSCodeSuggestion(
            code=r["code"],
            description=r["description"],
            confidence=r.get("similarity", 0.5),
            category=r["code"][0] if r["code"] else None
        )
        for r in hcpcs_results
    ]
    
    # Determine best procedure match
    best_procedure_match = None
    if cpt_results and hcpcs_results:
        best_cpt = cpt_results[0] if cpt_results else None
        best_hcpcs = hcpcs_results[0] if hcpcs_results else None
        
        if best_cpt and best_hcpcs:
            if best_cpt.get("similarity", 0) >= best_hcpcs.get("similarity", 0):
                best_procedure_match = schemas.CodeSuggestion(
                    code=best_cpt["code"],
                    description=best_cpt["description"],
                    confidence=best_cpt.get("similarity", 0.5),
                    code_type="CPT"
                )
            else:
                best_procedure_match = schemas.CodeSuggestion(
                    code=best_hcpcs["code"],
                    description=best_hcpcs["description"],
                    confidence=best_hcpcs.get("similarity", 0.5),
                    code_type="HCPCS"
                )
    elif cpt_results:
        best_procedure_match = schemas.CodeSuggestion(
            code=cpt_results[0]["code"],
            description=cpt_results[0]["description"],
            confidence=cpt_results[0].get("similarity", 0.5),
            code_type="CPT"
        )
    elif hcpcs_results:
        best_procedure_match = schemas.CodeSuggestion(
            code=hcpcs_results[0]["code"],
            description=hcpcs_results[0]["description"],
            confidence=hcpcs_results[0].get("similarity", 0.5),
            code_type="HCPCS"
        )
    
    return schemas.EnhancedCodeSuggestionResponse(
        icd_suggestions=icd_suggestions,
        cpt_suggestions=cpt_suggestions,
        hcpcs_suggestions=hcpcs_suggestions,
        best_procedure_match=best_procedure_match,
        search_method="enhanced_semantic_search",
        notes_analyzed=analysis_context[:100] + "..." if len(analysis_context) > 100 else analysis_context
    )

# Rest of the existing endpoints remain the same...
# (UPDATE CLAIM, DELETE CLAIM, CLAIM STATISTICS, SERVICE LINES MANAGEMENT, etc.)

# -------------------- UPDATE CLAIM --------------------
@router.put("/{claim_id}", response_model=schemas.ClaimOut)
def update_claim(
    claim_id: UUID,
    claim_update: schemas.ClaimUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing claim"""
    claim = db.query(models.Claim).filter(models.Claim.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    # Update claim fields
    update_data = claim_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(claim, field, value)

    db.commit()
    db.refresh(claim)

    return get_claim(claim_id, db)

# -------------------- DELETE CLAIM --------------------
@router.delete("/{claim_id}")
def delete_claim(claim_id: UUID, db: Session = Depends(get_db)):
    """Delete a claim and all related data"""
    claim = db.query(models.Claim).filter(models.Claim.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    db.delete(claim)
    db.commit()

    return {"message": "Claim deleted successfully"}

# -------------------- CLAIM STATISTICS --------------------
@router.get("/statistics/summary", response_model=schemas.ClaimSummaryResponse)
def get_claim_statistics(db: Session = Depends(get_db)):
    """Get claim statistics summary"""
    from sqlalchemy import func
    
    total_claims = db.query(func.count(models.Claim.claim_id)).scalar()
    
    total_amount = db.query(func.sum(models.Claim.total_charge)).scalar() or 0
    
    pending_claims = db.query(func.count(models.Claim.claim_id)).filter(
        models.Claim.claim_status.in_(["DRAFT", "SUBMITTED", "IN_PROCESS"])
    ).scalar()
    
    approved_claims = db.query(func.count(models.Claim.claim_id)).filter(
        models.Claim.claim_status.in_(["APPROVED", "PAID"])
    ).scalar()
    
    denied_claims = db.query(func.count(models.Claim.claim_id)).filter(
        models.Claim.claim_status.in_(["DENIED", "REJECTED"])
    ).scalar()

    return schemas.ClaimSummaryResponse(
        total_claims=total_claims,
        total_amount=total_amount,
        pending_claims=pending_claims,
        approved_claims=approved_claims,
        denied_claims=denied_claims
    )

# -------------------- SERVICE LINES MANAGEMENT --------------------
@router.post("/{claim_id}/service-lines", response_model=schemas.ClaimServiceLineOut)
def add_service_line(
    claim_id: UUID,
    service_line: schemas.ClaimServiceLineCreate,
    db: Session = Depends(get_db)
):
    """Add a service line to an existing claim"""
    claim = db.query(models.Claim).filter(models.Claim.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    service_line_dict = service_line.dict()
    
    # Ensure code_type is set
    if 'code_type' not in service_line_dict or not service_line_dict['code_type']:
        code = service_line_dict.get('cpt_hcpcs_code', '')
        if code and len(code) >= 3:
            if code[0].isalpha():
                service_line_dict['code_type'] = 'HCPCS'
            else:
                service_line_dict['code_type'] = 'CPT'
        else:
            service_line_dict['code_type'] = 'CPT'

    new_service_line = models.ClaimServiceLine(
        claim_id=claim_id,
        **service_line_dict
    )
    db.add(new_service_line)
    db.commit()
    db.refresh(new_service_line)

    return new_service_line