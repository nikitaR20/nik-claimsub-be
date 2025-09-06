import uuid
from sqlalchemy import Column, String, Text, Boolean, Date, DateTime, Float, ForeignKey, TIMESTAMP, LargeBinary,MetaData,Table
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base

class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column("PatientID", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column("FirstName", String)
    last_name = Column("LastName", String)
    patient_age = Column("PatientAge", Float)
    patient_gender = Column("PatientGender", String)
    patient_income = Column("PatientIncome", Float)
    patient_marital_status = Column("PatientMaritalStatus", String)
    patient_employment_status = Column("PatientEmploymentStatus", String)

    claims = relationship("Claim", back_populates="patient")


class Provider(Base):
    __tablename__ = "providers"

    provider_id = Column("ProviderID", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column("ProviderFirstName", String(100))
    last_name = Column("ProviderLastName", String(100))
    specialty = Column("ProviderSpeciality", String(100))
    location = Column("ProviderLocation", String(255))

    claims = relationship("Claim", back_populates="provider")


class Claim(Base):
    __tablename__ = "claims"

    claim_id = Column("ClaimID", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column("PatientID", PG_UUID(as_uuid=True), ForeignKey("patients.PatientID"), nullable=False)
    provider_id = Column("ProviderID", PG_UUID(as_uuid=True), ForeignKey("providers.ProviderID"), nullable=False)
    claim_amount = Column("ClaimAmount", Float, default=0)
    claim_date = Column("ClaimDate", Date)
    diagnosis_code = Column("DiagnosisCode", String)
    procedure_code = Column("ProcedureCode", String)
    claim_status = Column("ClaimStatus", String(50), default="Pending")
    claim_type = Column("ClaimType", String(50))
    claim_submission_method = Column("ClaimSubmissionMethod", String(50))
    predicted_payout = Column("PredictedPayout", Float)
    approval_probability = Column("ApprovalProbability", Float)
    coverage_notes = Column("CoverageNotes", Text)
    suggested_diagnosis_code = Column("SuggestedDiagnosisCode", String(50))
    suggested_procedure_code = Column("SuggestedProcedureCode", String(50))
    fraud_flag = Column("FraudFlag", Boolean)
    fraud_reason = Column("FraudReason", Text)
    created_at = Column("CreatedAt", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("UpdatedAt", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    patient = relationship("Patient", back_populates="claims")
    provider = relationship("Provider", back_populates="claims")
    documents = relationship("ClaimDocument", back_populates="claim")


# -------------------- ClaimDocument --------------------
class ClaimDocument(Base):
    __tablename__ = "claim_documents"

    document_id = Column("document_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column("ClaimID", PG_UUID(as_uuid=True), ForeignKey("claims.ClaimID"), nullable=False)
    document_type = Column("document_type", String, nullable=False)
    file_name = Column("file_name", String, nullable=False)
    content_type = Column("content_type", String)
    description = Column("description", Text)
    file_data = Column("file_data", LargeBinary)
    ocr_redacted_text = Column("ocr_redacted_text", Text)
    uploaded_at = Column("uploaded_at", DateTime(timezone=True), server_default=func.now())

    claim = relationship("Claim", back_populates="documents")

    # ICD-10 table
icd10 = Table(
    "icd10",
    Base.metadata,
    Column("code", String, primary_key=True),
    Column("description", String),
)

    # Procedure codes table
procedure_codes = Table(
    "procedure_codes",
    Base.metadata,
    Column("code", String, primary_key=True),
    Column("description", String),
)