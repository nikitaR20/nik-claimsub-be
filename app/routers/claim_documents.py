# app/routers/claim_documents.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from datetime import datetime
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/claim-documents", tags=["Claim Documents"])

# ----------------- GET ALL DOCUMENTS -----------------
@router.get("/", response_model=List[schemas.ClaimDocumentResponse])
def get_claim_documents(db: Session = Depends(get_db)):
    return db.query(models.ClaimDocument).all()

# ------------- GET DOCUMENTS BY CLAIM ID -------------
@router.get("/{claim_id}", response_model=List[schemas.ClaimDocumentResponse])
def get_documents_by_claim(claim_id: str, db: Session = Depends(get_db)):
    # Convert to UUID
    try:
        claim_uuid = UUID(claim_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid claim_id format")

    # Query documents
    docs = db.query(models.ClaimDocument).filter(models.ClaimDocument.claim_id == claim_uuid).all()
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found for this claim")
    return docs

# ------------------- UPLOAD DOCUMENT -------------------
@router.post("/upload", response_model=schemas.ClaimDocumentResponse)
async def upload_document(
    claim_id: str = Form(...),
    document_type: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Convert claim_id to UUID
    try:
        claim_uuid = UUID(claim_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid claim_id format")

    # Read file bytes
    file_data = await file.read()

    # Create new document
    new_doc = models.ClaimDocument(
        claim_id=claim_uuid,
        document_type=document_type,
        file_name=file.filename,
        content_type=file.content_type,
        description=description,
        file_data=file_data,
        uploaded_at=datetime.utcnow()
    )

    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    return new_doc
