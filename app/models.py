import uuid
from sqlalchemy import Column, String, Text, Boolean, Date, DateTime, Float, ForeignKey, TIMESTAMP, LargeBinary, MetaData, Table, Integer, Numeric, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base
from passlib.context import CryptContext

class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column("patient_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column("first_name", String(100), nullable=False)
    last_name = Column("last_name", String(100), nullable=False)
    middle_initial = Column("middle_initial", String(10))
    birth_date = Column("birth_date", Date, nullable=False)
    gender = Column("gender", String(1), CheckConstraint("gender IN ('M', 'F', 'X')"), nullable=False)
    age = Column("age", Integer)
    
    # Address Information
    address_line1 = Column("address_line1", String(255))
    address_line2 = Column("address_line2", String(255))
    city = Column("city", String(100))
    state = Column("state", String(2))
    zip_code = Column("zip_code", String(10))
    phone = Column("phone", String(20))
    email = Column("email", String(255))
    
    # Additional Demographics
    income = Column("income", Numeric(12, 2))
    marital_status = Column("marital_status", String(20))
    employment_status = Column("employment_status", String(50))
    relationship_to_insured = Column("relationship_to_insured", String(20), 
                                   CheckConstraint("relationship_to_insured IN ('SELF', 'SPOUSE', 'CHILD', 'OTHER')"))
    
    # Emergency Contact
    emergency_contact_name = Column("emergency_contact_name", String(255))
    emergency_contact_phone = Column("emergency_contact_phone", String(20))
    
    # Timestamps
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    claims = relationship("Claim", back_populates="patient")
    insurances = relationship("PatientInsurance", back_populates="patient")


class Provider(Base):
    __tablename__ = "providers"

    provider_id = Column("provider_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column("first_name", String(100), nullable=False)
    last_name = Column("last_name", String(100), nullable=False)
    middle_initial = Column("middle_initial", String(10))
    credentials = Column("credentials", String(50))
    npi = Column("npi", String(20), nullable=False, unique=True)
    tax_id = Column("tax_id", String(20))
    specialty = Column("specialty", String(100))
    
    # Address Information
    address_line1 = Column("address_line1", String(255))
    address_line2 = Column("address_line2", String(255))
    city = Column("city", String(100))
    state = Column("state", String(2))
    zip_code = Column("zip_code", String(10))
    phone = Column("phone", String(20))
    email = Column("email", String(255))
    fax = Column("fax", String(20))
    
    # Professional Information
    taxonomy_code = Column("taxonomy_code", String(20))
    license_number = Column("license_number", String(50))
    license_state = Column("license_state", String(2))
    location = Column("location", String(255))
    
    # Timestamps
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    claims = relationship("Claim", back_populates="provider")


class PatientInsurance(Base):
    __tablename__ = "patient_insurances"

    insurance_id = Column("insurance_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column("patient_id", PG_UUID(as_uuid=True), ForeignKey("patients.patient_id"), nullable=False)
    
    # Box 1 - Insurance Type
    insurance_type = Column("insurance_type", String(20), 
                           CheckConstraint("insurance_type IN ('MEDICARE', 'MEDICAID', 'TRICARE', 'CHAMPVA', 'GROUP_HEALTH', 'FECA', 'OTHER')"),
                           nullable=False)
    
    # Box 1a, 11 - Policy Information
    policy_number = Column("policy_number", String(50))
    group_number = Column("group_number", String(50))
    plan_name = Column("plan_name", String(255))
    
    # Box 4, 7, 11a - Insured Information
    insured_name = Column("insured_name", String(255))
    insured_dob = Column("insured_dob", Date)
    insured_sex = Column("insured_sex", String(1), CheckConstraint("insured_sex IN ('M', 'F', 'X')"))
    relationship_to_patient = Column("relationship_to_patient", String(20),
                                    CheckConstraint("relationship_to_patient IN ('SELF', 'SPOUSE', 'CHILD', 'OTHER')"))
    
    # Insured Address
    address_line1 = Column("address_line1", String(255))
    address_line2 = Column("address_line2", String(255))
    city = Column("city", String(100))
    state = Column("state", String(2))
    zip_code = Column("zip_code", String(10))
    phone = Column("phone", String(20))
    
    # Box 9, 9a, 9d - Secondary Insurance
    other_insured_name = Column("other_insured_name", String(255))
    other_insured_policy_number = Column("other_insured_policy_number", String(50))
    other_insured_group_number = Column("other_insured_group_number", String(50))
    other_insured_dob = Column("other_insured_dob", Date)
    other_insured_sex = Column("other_insured_sex", String(1), CheckConstraint("other_insured_sex IN ('M', 'F', 'X')"))
    other_insured_employer = Column("other_insured_employer", String(255))
    other_insured_plan_name = Column("other_insured_plan_name", String(255))
    
    # Box 10 - Condition Related
    condition_employment_related = Column("condition_employment_related", Boolean, default=False)
    condition_auto_accident = Column("condition_auto_accident", Boolean, default=False)
    condition_other_accident = Column("condition_other_accident", Boolean, default=False)
    
    # Coverage Dates
    effective_date = Column("effective_date", Date)
    termination_date = Column("termination_date", Date)
    
    # Timestamps
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="insurances")


class Claim(Base):
    __tablename__ = "claims"

    claim_id = Column("claim_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column("patient_id", PG_UUID(as_uuid=True), ForeignKey("patients.patient_id"), nullable=False)
    provider_id = Column("provider_id", PG_UUID(as_uuid=True), ForeignKey("providers.provider_id"), nullable=False)
    insurance_id = Column("insurance_id", PG_UUID(as_uuid=True), ForeignKey("patient_insurances.insurance_id"))
    
    # Claim Identification
    claim_number = Column("claim_number", String(50), unique=True)
    claim_date = Column("claim_date", Date, nullable=False)
    claim_status = Column("claim_status", String(50), 
                         CheckConstraint("claim_status IN ('DRAFT', 'SUBMITTED', 'IN_PROCESS', 'APPROVED', 'DENIED', 'PAID', 'REJECTED')"),
                         default="DRAFT")
    claim_type = Column("claim_type", String(50))
    submission_method = Column("submission_method", String(50))
    
    # Financial Information
    claim_amount = Column("claim_amount", Numeric(12, 2))
    total_charge = Column("total_charge", Numeric(12, 2))
    amount_paid = Column("amount_paid", Numeric(12, 2))
    balance_due = Column("balance_due", Numeric(12, 2))
    
    # Box 10 - Condition Related (expanded)
    condition_employment_related = Column("condition_employment_related", Boolean, default=False)
    condition_auto_accident = Column("condition_auto_accident", Boolean, default=False)
    condition_auto_accident_state = Column("condition_auto_accident_state", String(2))
    condition_other_accident = Column("condition_other_accident", Boolean, default=False)
    condition_claim_codes = Column("condition_claim_codes", String(100))
    
    # Box 14-19 - Additional Dates and References
    date_current_illness = Column("date_current_illness", Date)
    date_first_consultation = Column("date_first_consultation", Date)
    date_similar_illness = Column("date_similar_illness", Date)
    dates_unable_to_work_from = Column("dates_unable_to_work_from", Date)
    dates_unable_to_work_to = Column("dates_unable_to_work_to", Date)
    referring_provider_name = Column("referring_provider_name", String(255))
    referring_provider_npi = Column("referring_provider_npi", String(20))
    referring_provider_other_id = Column("referring_provider_other_id", String(50))
    hospitalization_date_from = Column("hospitalization_date_from", Date)
    hospitalization_date_to = Column("hospitalization_date_to", Date)
    additional_claim_info = Column("additional_claim_info", Text)
    
    # Box 20 - Outside Lab
    outside_lab = Column("outside_lab", Boolean, default=False)
    outside_lab_charges = Column("outside_lab_charges", Numeric(10, 2))
    
    # Box 22 - Resubmission
    resubmission_code = Column("resubmission_code", String(10))
    original_reference_number = Column("original_reference_number", String(50))
    
    # Box 23 - Authorization
    authorization_number = Column("authorization_number", String(50))
    
    # Box 25-27 - Provider Information
    tax_id_type = Column("tax_id_type", String(10), CheckConstraint("tax_id_type IN ('EIN', 'SSN')"))
    patient_account_number = Column("patient_account_number", String(50))
    accept_assignment = Column("accept_assignment", Boolean)
    
    # Box 31 - Signatures
    physician_signature_date = Column("physician_signature_date", Date)
    physician_signature_on_file = Column("physician_signature_on_file", Boolean, default=False)
    
    # Box 32 - Service Facility
    service_facility_name = Column("service_facility_name", String(255))
    service_facility_address_line1 = Column("service_facility_address_line1", String(255))
    service_facility_address_line2 = Column("service_facility_address_line2", String(255))
    service_facility_city = Column("service_facility_city", String(100))
    service_facility_state = Column("service_facility_state", String(2))
    service_facility_zip_code = Column("service_facility_zip_code", String(10))
    service_facility_npi = Column("service_facility_npi", String(20))
    service_facility_other_id = Column("service_facility_other_id", String(50))
    
    # Box 33 - Billing Provider
    billing_provider_name = Column("billing_provider_name", String(255))
    billing_provider_address_line1 = Column("billing_provider_address_line1", String(255))
    billing_provider_address_line2 = Column("billing_provider_address_line2", String(255))
    billing_provider_city = Column("billing_provider_city", String(100))
    billing_provider_state = Column("billing_provider_state", String(2))
    billing_provider_zip_code = Column("billing_provider_zip_code", String(10))
    billing_provider_phone = Column("billing_provider_phone", String(20))
    billing_provider_npi = Column("billing_provider_npi", String(20))
    billing_provider_other_id = Column("billing_provider_other_id", String(50))
    
    # Additional Tracking
    submission_date = Column("submission_date", Date)
    received_date = Column("received_date", Date)
    processed_date = Column("processed_date", Date)
    paid_date = Column("paid_date", Date)
    claim_frequency_code = Column("claim_frequency_code", String(2), default='1')
    medical_record_number = Column("medical_record_number", String(50))
    demonstration_project_id = Column("demonstration_project_id", String(50))
    
    # Legacy/Analysis Fields (keeping for backward compatibility)
    primary_diagnosis_code = Column("primary_diagnosis_code", String(10))
    primary_procedure_code = Column("primary_procedure_code", String(10))
    predicted_payout = Column("predicted_payout", Numeric(12, 2))
    approval_probability = Column("approval_probability", Numeric(5, 4))
    fraud_flag = Column("fraud_flag", Boolean, default=False)
    fraud_reason = Column("fraud_reason", Text)
    coverage_notes = Column("coverage_notes", Text)
    suggested_diagnosis_code = Column("suggested_diagnosis_code", String(10))
    suggested_procedure_code = Column("suggested_procedure_code", String(10))
    
    # Timestamps
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="claims")
    provider = relationship("Provider", back_populates="claims")
    insurance = relationship("PatientInsurance")
    service_lines = relationship("ClaimServiceLine", back_populates="claim", cascade="all, delete-orphan")
    diagnoses = relationship("ClaimDiagnosis", back_populates="claim", cascade="all, delete-orphan")
    documents = relationship("ClaimDocument", back_populates="claim", cascade="all, delete-orphan")


class ClaimServiceLine(Base):
    __tablename__ = "claim_service_lines"

    service_line_id = Column("service_line_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column("claim_id", PG_UUID(as_uuid=True), ForeignKey("claims.claim_id"), nullable=False)
    
    # Box 24A - Service Dates
    line_number = Column("line_number", Integer)
    service_date_from = Column("service_date_from", Date, nullable=False)
    service_date_to = Column("service_date_to", Date)
    
    # Box 24B-C
    place_of_service_code = Column("place_of_service_code", String(2), nullable=False)
    emergency_indicator = Column("emergency_indicator", String(1))
    
    # Box 24D - Procedures and Modifiers (updated for HCPCS)
    cpt_hcpcs_code = Column("cpt_hcpcs_code", String(10), nullable=False)  # Can be CPT or HCPCS
    code_type = Column("code_type", String(10), default="CPT")  # CPT or HCPCS
    modifier1 = Column("modifier1", String(2))
    modifier2 = Column("modifier2", String(2))
    modifier3 = Column("modifier3", String(2))
    modifier4 = Column("modifier4", String(2))
    
    # Box 24E - Diagnosis Pointer
    diagnosis_pointer = Column("diagnosis_pointer", String(12))
    
    # Box 24F-G - Charges and Units
    charge_amount = Column("charge_amount", Numeric(10, 2), nullable=False)
    units = Column("units", Integer, default=1)
    
    # Box 24H-J - Additional Information
    epsdt_indicator = Column("epsdt_indicator", String(1))
    rendering_provider_npi = Column("rendering_provider_npi", String(20))
    rendering_provider_other_id = Column("rendering_provider_other_id", String(50))
    rendering_provider_qualifier = Column("rendering_provider_qualifier", String(2))
    revenue_code = Column("revenue_code", String(10))
    
    # Timestamps
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    claim = relationship("Claim", back_populates="service_lines")


class ClaimDiagnosis(Base):
    __tablename__ = "claim_diagnoses"

    claim_diagnosis_id = Column("claim_diagnosis_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column("claim_id", PG_UUID(as_uuid=True), ForeignKey("claims.claim_id"), nullable=False)
    
    # Box 21 - Diagnosis Information
    position = Column("position", Integer, nullable=False)  # 1-12 for A-L
    icd10_code = Column("icd10_code", String(10), nullable=False)
    diagnosis_pointer = Column("diagnosis_pointer", String(1), CheckConstraint("diagnosis_pointer IN ('A','B','C','D','E','F','G','H','I','J','K','L')"))
    
    # Timestamps
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    claim = relationship("Claim", back_populates="diagnoses")


class ClaimDocument(Base):
    __tablename__ = "claim_documents"

    document_id = Column("document_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column("claim_id", PG_UUID(as_uuid=True), ForeignKey("claims.claim_id"), nullable=False)
    document_type = Column("document_type", String(50), nullable=False)
    file_name = Column("file_name", String(255), nullable=False)
    content_type = Column("content_type", String(100))
    description = Column("description", Text)
    file_data = Column("file_data", LargeBinary)
    ocr_redacted_text = Column("ocr_redacted_text", Text)
    uploaded_at = Column("uploaded_at", TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    claim = relationship("Claim", back_populates="documents")


class Segment(Base):
    __tablename__ = "segments"
    
    segment_id = Column("segment_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column("name", String(255), nullable=False)
    description = Column("description", Text)
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


# Reference Tables
icd10 = Table(
    "icd10",
    Base.metadata,
    Column("code", String, primary_key=True),
    Column("description", String),
)
hcpcs_codes = Table(
    "hcpcs_codes",
    Base.metadata,
    Column("code", String, primary_key=True),
    Column("description", String),
    Column("embedding", Text),  # For AI/ML features
)

icd10_codes = Table(
    "icd10_codes",
    Base.metadata,
    Column("code", String, primary_key=True),
    Column("description", String),
    Column("embedding", Text),  # For AI/ML features
)

procedure_codes = Table(
    "procedure_codes",
    Base.metadata,
    Column("code", String, primary_key=True),
    Column("description", String),
)

procedural_codes = Table(
    "procedural_codes",
    Base.metadata,
    Column("code", String, primary_key=True),
    Column("description", String),
    Column("embedding", Text),  # For AI/ML features
)


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"

    user_id = Column("user_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column("username", String(50), unique=True, nullable=False, index=True)
    email = Column("email", String(255), unique=True, nullable=False, index=True)
    hashed_password = Column("hashed_password", String(255), nullable=False)
    
    # Personal Information
    first_name = Column("first_name", String(100), nullable=False)
    last_name = Column("last_name", String(100), nullable=False)
    middle_initial = Column("middle_initial", String(10))
    
    # Role and Status
    role = Column("role", String(20), 
             CheckConstraint("role IN ('ADMIN', 'MEDICAL_STAFF', 'INSURANCE_STAFF', 'BILLING_STAFF')"), 
             nullable=False)
    is_active = Column("is_active", Boolean, default=True)
    is_verified = Column("is_verified", Boolean, default=False)
    
    # Contact Information
    phone = Column("phone", String(20))
    address_line1 = Column("address_line1", String(255))
    address_line2 = Column("address_line2", String(255))
    city = Column("city", String(100))
    state = Column("state", String(2))
    zip_code = Column("zip_code", String(10))
    
    # Professional Information (for providers)
    npi = Column("npi", String(20), unique=True, index=True)  # National Provider Identifier
    license_number = Column("license_number", String(50))
    specialty = Column("specialty", String(100))
    organization = Column("organization", String(255))  # Hospital/Clinic name
    
    # Insurance Staff Information
    insurance_company = Column("insurance_company", String(255))
    
    # Access Control
    last_login = Column("last_login", TIMESTAMP(timezone=True))
    login_attempts = Column("login_attempts", Integer, default=0)
    locked_until = Column("locked_until", TIMESTAMP(timezone=True))
    
    # Password Reset
    reset_token = Column("reset_token", String(255))
    reset_token_expires = Column("reset_token_expires", TIMESTAMP(timezone=True))
    
    # Timestamps
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column("created_by", PG_UUID(as_uuid=True), ForeignKey("users.user_id"))
    
    # Relationships
    created_by_user = relationship("User", remote_side=[user_id])
    user_permissions = relationship("UserPermission", back_populates="user", cascade="all, delete-orphan",foreign_keys="UserPermission.user_id")
    audit_logs = relationship("AuditLog", back_populates="user")

    def verify_password(self, password: str) -> bool:
        """Verify a password against the hash"""
        return pwd_context.verify(password, self.hashed_password)
    
    def set_password(self, password: str):
        """Hash and set a new password"""
        self.hashed_password = pwd_context.hash(password)
    
    @property
    def full_name(self):
        """Get user's full name"""
        middle = f" {self.middle_initial}" if self.middle_initial else ""
        return f"{self.first_name}{middle} {self.last_name}"


class Permission(Base):
    __tablename__ = "permissions"
    
    permission_id = Column("permission_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column("name", String(100), unique=True, nullable=False)
    description = Column("description", Text)
    resource = Column("resource", String(50), nullable=False)  # patients, claims, providers, etc.
    action = Column("action", String(50), nullable=False)  # create, read, update, delete, list
    
    # Timestamps
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user_permissions = relationship("UserPermission", back_populates="permission")


class UserPermission(Base):
    __tablename__ = "user_permissions"
    
    user_permission_id = Column("user_permission_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column("user_id", PG_UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    permission_id = Column("permission_id", PG_UUID(as_uuid=True), ForeignKey("permissions.permission_id"), nullable=False)
    
    # Optional constraints (can restrict access further)
    resource_id = Column("resource_id", String(255))  # Specific resource ID if needed
    conditions = Column("conditions", Text)  # JSON conditions for complex permissions
    
    # Timestamps
    granted_at = Column("granted_at", TIMESTAMP(timezone=True), server_default=func.now())
    granted_by = Column("granted_by", PG_UUID(as_uuid=True), ForeignKey("users.user_id"))
    expires_at = Column("expires_at", TIMESTAMP(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="user_permissions", foreign_keys=[user_id])
    permission = relationship("Permission", back_populates="user_permissions")
    granted_by_user = relationship("User", foreign_keys=[granted_by])


class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    audit_id = Column("audit_id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column("user_id", PG_UUID(as_uuid=True), ForeignKey("users.user_id"))
    
    # Action Information
    action = Column("action", String(50), nullable=False)  # LOGIN, CREATE_CLAIM, UPDATE_PATIENT, etc.
    resource_type = Column("resource_type", String(50))  # claims, patients, providers
    resource_id = Column("resource_id", String(255))  # ID of the affected resource
    
    # Request Information
    endpoint = Column("endpoint", String(255))
    method = Column("method", String(10))
    ip_address = Column("ip_address", String(45))
    user_agent = Column("user_agent", Text)
    
    # Details
    description = Column("description", Text)
    old_values = Column("old_values", Text)  # JSON of old values for updates
    new_values = Column("new_values", Text)  # JSON of new values for updates
    
    # Status
    status = Column("status", String(20), default="SUCCESS")  # SUCCESS, FAILED, ERROR
    error_message = Column("error_message", Text)
    
    # Timestamp
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")


class UserSession(Base):
    __tablename__ = "user_sessions"
    
    session_id = Column("session_id", String(255), primary_key=True)
    user_id = Column("user_id", PG_UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    
    # Session Information
    ip_address = Column("ip_address", String(45))
    user_agent = Column("user_agent", Text)
    is_active = Column("is_active", Boolean, default=True)
    
    # Timestamps
    created_at = Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())