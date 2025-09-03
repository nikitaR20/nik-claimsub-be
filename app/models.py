import uuid
from sqlalchemy import Column, String, Text, Boolean, Date, DateTime, LargeBinary, ForeignKey, Integer,UUID,Float,TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.sql import func
from app.database import Base
from sqlalchemy.orm import relationship
class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column("PatientID", UUID(as_uuid=True), primary_key=True)
    first_name = Column("FirstName", String)
    last_name = Column("LastName", String)
    age = Column("PatientAge", Integer)
    gender = Column("PatientGender", String)
    income = Column("PatientIncome", Float)
    marital_status = Column("PatientMaritalStatus", String)
    employment_status = Column("PatientEmploymentStatus", String)
    claims = relationship("Claim", back_populates="patient")

class Provider(Base):
    __tablename__ = "providers"

    provider_id = Column("ProviderID",UUID(as_uuid=True), primary_key=True)
    first_name = Column("ProviderFirstName", String(100))
    last_name = Column("ProviderLastName", String(100))
    specialty = Column("ProviderSpeciality", String(100))
    location = Column("ProviderLocation", String(255))
    # Relationship to Claims
    claims = relationship("Claim", back_populates="provider")
    #created_at = Column(DateTime(timezone=True), server_default=func.now())
    #updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Segment(Base):
    __tablename__ = "segments"

    segment_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Claim(Base):
    __tablename__ = "claims"

    claim_id = Column("ClaimID", UUID(as_uuid=True), primary_key=True)
    #patient_id = Column("PatientID", UUID(as_uuid=True))
    #provider_id = Column("ProviderID", UUID(as_uuid=True))
    patient_id = Column("PatientID", UUID(as_uuid=True), ForeignKey("patients.PatientID"), nullable=False)
    provider_id = Column("ProviderID", UUID(as_uuid=True), ForeignKey("providers.ProviderID"), nullable=False)
    claim_amount = Column("ClaimAmount", Float)
    claim_date = Column("ClaimDate", Date)
    diagnosis_code = Column("DiagnosisCode", String)
    procedure_code = Column("ProcedureCode", String)
    #patient_age = Column("PatientAge", Float)
    #patient_gender = Column("PatientGender", String(10))
    #provider_specialty = Column("ProviderSpeciality", String(255))
    claim_status = Column("ClaimStatus", String(50))
    #patient_income = Column("PatientIncome", Float)
    #patient_marital_status = Column("PatientMaritalStatus", String(20))
    #patient_employment_status = Column("PatientEmploymentStatus", String(50))
    #provider_location = Column("ProviderLocation", String(255))
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
    
class ClaimDocument(Base):
    __tablename__ = "claim_documents"

    document_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(PG_UUID(as_uuid=True), ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False)
    document_type = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    file_data = Column(LargeBinary, nullable=False)
    content_type = Column(String, nullable=False)
    description = Column(Text)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    ocr_redacted_text = Column(Text)
