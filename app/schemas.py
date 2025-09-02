from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import date, datetime

class PatientOut(BaseModel):
    patient_id: UUID
    first_name: Optional[str]
    last_name: Optional[str]
    patient_age: Optional[float]        # ✅ float
    patient_gender: Optional[str]
    patient_income: Optional[float]     # ✅ float
    patient_marital_status: Optional[str]
    patient_employment_status: Optional[str]

    class Config:
        orm_mode = True

class ProviderOut(BaseModel):
    provider_id: UUID
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    specialty: Optional[str] = None
    location: Optional[str] = None

    class Config:
        orm_mode = True

class SegmentOut(BaseModel):
    segment_id: UUID
    name: str
    description: Optional[str] = None

    class Config:
        orm_mode = True

class ClaimOut(BaseModel):
    claim_id: UUID
    patient_id: UUID
    provider_id: UUID
    claim_amount: Optional[float]       # ✅ float
    claim_date: Optional[date]
    diagnosis_code: Optional[str]
    procedure_code: Optional[str]
    patient_age: Optional[float]        # ✅ float
    patient_gender: Optional[str]
    provider_specialty: Optional[str]
    claim_status: Optional[str]
    patient_income: Optional[float]     # ✅ float
    patient_marital_status: Optional[str]
    patient_employment_status: Optional[str]
    provider_location: Optional[str]
    claim_type: Optional[str]
    claim_submission_method: Optional[str]
    predicted_payout: Optional[float]   # ✅ float
    approval_probability: Optional[float] # ✅ float
    coverage_notes: Optional[str]
    suggested_diagnosis_code: Optional[str]
    suggested_procedure_code: Optional[str]
    fraud_flag: Optional[bool]
    fraud_reason: Optional[str]

    class Config:
        orm_mode = True

class ClaimDocumentResponse(BaseModel):
    document_id: UUID
    claim_id: UUID
    document_type: str
    file_name: str
    content_type: str
    description: Optional[str]
    uploaded_at: datetime
    ocr_redacted_text: Optional[str]

    class Config:
        orm_mode = True
