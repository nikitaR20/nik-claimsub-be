from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List
from app import models, schemas
from app.database import get_db
from app.utils.ocr_pii import ocr_extract_text, redact_pii


router = APIRouter(
    prefix="/claim-documents",
    tags=["claim-documents"],
)

@router.get("/{claim_id}", response_model=List[schemas.ClaimDocumentResponse])
def get_documents_for_claim(
    claim_id: UUID,
    db: Session = Depends(get_db)
):
    documents = db.query(models.ClaimDocument).filter(models.ClaimDocument.claim_id == claim_id).all()
    return documents or []

@router.post("/upload", response_model=schemas.ClaimDocumentResponse)
async def upload_claim_document(
    claim_id: UUID = Form(...),
    document_type: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if file.content_type not in ["application/pdf", "image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed types: PDF, JPEG, PNG.")

    file_data = await file.read()
    if len(file_data) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")

    claim = db.query(models.Claim).filter(models.Claim.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found.")

    # OCR extraction
    extracted_text = ocr_extract_text(file_data, file.content_type)

    # PII redaction
    redacted_text = redact_pii(extracted_text)

    new_doc = models.ClaimDocument(
        claim_id=claim_id,
        document_type=document_type,
        file_name=file.filename,
        file_data=file_data,
        content_type=file.content_type,
        description=description,
        ocr_redacted_text=redacted_text
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    return new_doc
