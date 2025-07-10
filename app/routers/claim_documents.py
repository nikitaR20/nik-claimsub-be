from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID
from app import models, schemas
from app.database import get_db
from typing import List

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
    # Validate file type
    if file.content_type not in ["application/pdf", "image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed types: PDF, JPEG, PNG.")

    file_data = await file.read()
    if len(file_data) > 10 * 1024 * 1024:  # 10MB file size limit
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")

    # Check if claim exists
    claim = db.query(models.Claim).filter(models.Claim.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found.")

    new_doc = models.ClaimDocument(
        claim_id=claim_id,
        document_type=document_type,
        file_name=file.filename,
        file_data=file_data,
        content_type=file.content_type,
        description=description
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    return new_doc
