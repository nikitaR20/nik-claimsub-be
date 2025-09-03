from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import date, datetime

# -------------------- Patient Schema --------------------
class PatientOut(BaseModel):
    patient_id: UUID
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    patient_age: Optional[float] = None
    patient_gender: Optional[str] = None
    patient_income: Optional[float] = None
    patient_marital_status: Optional[str] = None
    patient_employment_status: Optional[str] = None

    class Config:
        orm_mode = True


# -------------------- Provider Schema --------------------
class ProviderOut(BaseModel):
    provider_id: UUID
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    specialty: Optional[str] = None
    location: Optional[str] = None

    class Config:
        orm_mode = True


# -------------------- Segment Schema --------------------
class SegmentOut(BaseModel):
    segment_id: UUID
    name: str
    description: Optional[str] = None

    class Config:
        orm_mode = True


# -------------------- Claim Schema --------------------
class ClaimOut(BaseModel):
    claim_id: UUID
    patient_id: UUID
    provider_id: UUID
    claim_amount: Optional[float] = None
    claim_date: Optional[date] = None
    diagnosis_code: Optional[str] = None
    procedure_code: Optional[str] = None
    patient_age: Optional[float] = None
    patient_gender: Optional[str] = None
    provider_specialty: Optional[str] = None
    claim_status: Optional[str] = None
    patient_income: Optional[float] = None
    patient_marital_status: Optional[str] = None
    patient_employment_status: Optional[str] = None
    provider_location: Optional[str] = None
    claim_type: Optional[str] = None
    claim_submission_method: Optional[str] = None
    predicted_payout: Optional[float] = None
    approval_probability: Optional[float] = None
    coverage_notes: Optional[str] = None
    suggested_diagnosis_code: Optional[str] = None
    suggested_procedure_code: Optional[str] = None
    fraud_flag: Optional[bool] = None
    fraud_reason: Optional[str] = None

    class Config:
        orm_mode = True


# -------------------- Claim Document Schema --------------------
class ClaimDocumentResponse(BaseModel):
    document_id: UUID
    claim_id: UUID
    document_type: str
    file_name: str
    content_type: str
    description: Optional[str] = None
    uploaded_at: datetime
    ocr_redacted_text: Optional[str] = None

    class Config:
        orm_mode = True
