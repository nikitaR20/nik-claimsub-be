from pydantic import BaseModel, Field, validator
from pydantic.networks import EmailStr
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

# Enums for validation
class GenderEnum(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "X"

class InsuranceTypeEnum(str, Enum):
    MEDICARE = "MEDICARE"
    MEDICAID = "MEDICAID"
    TRICARE = "TRICARE"
    CHAMPVA = "CHAMPVA"
    GROUP_HEALTH = "GROUP_HEALTH"
    FECA = "FECA"
    OTHER = "OTHER"

class RelationshipEnum(str, Enum):
    SELF = "SELF"
    SPOUSE = "SPOUSE"
    CHILD = "CHILD"
    OTHER = "OTHER"

class ClaimStatusEnum(str, Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    IN_PROCESS = "IN_PROCESS"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    PAID = "PAID"
    REJECTED = "REJECTED"
    PENDING = "PENDING"
class TaxIdTypeEnum(str, Enum):
    EIN = "EIN"
    SSN = "SSN"
    
class CodeTypeEnum(str, Enum):
    CPT = "CPT"
    HCPCS = "HCPCS"
# Base Models
class BaseModelWithTimestamps(BaseModel):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

# Segment Schema (keeping existing)
class SegmentOut(BaseModel):
    segment_id: UUID
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True

# Patient Models
class PatientBase(BaseModel):
    first_name: str = Field(..., max_length=100)
    last_name: str = Field(..., max_length=100)
    middle_initial: Optional[str] = Field(None, max_length=10)
    birth_date: Optional[date]
    gender: GenderEnum
    age: Optional[int] = None
    
    # Address Information
    address_line1: Optional[str] = Field(None, max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=2)
    zip_code: Optional[str] = Field(None, max_length=10)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=255)
    
    # Additional Demographics
    income: Optional[Decimal] = None
    marital_status: Optional[str] = Field(None, max_length=20)
    employment_status: Optional[str] = Field(None, max_length=50)
    relationship_to_insured: Optional[RelationshipEnum] = None
    
    # Emergency Contact
    emergency_contact_name: Optional[str] = Field(None, max_length=255)
    emergency_contact_phone: Optional[str] = Field(None, max_length=20)

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    middle_initial: Optional[str] = Field(None, max_length=10)
    birth_date: Optional[date] = None
    gender: Optional[GenderEnum] = None
    address_line1: Optional[str] = Field(None, max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=2)
    zip_code: Optional[str] = Field(None, max_length=10)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=255)
    income: Optional[Decimal] = None
    marital_status: Optional[str] = Field(None, max_length=20)
    employment_status: Optional[str] = Field(None, max_length=50)

class PatientOut(PatientBase, BaseModelWithTimestamps):
    patient_id: UUID
    
    class Config:
        from_attributes = True

# Provider Models
class ProviderBase(BaseModel):
    first_name: str = Field(..., max_length=100)
    last_name: str = Field(..., max_length=100)
    middle_initial: Optional[str] = Field(None, max_length=10)
    credentials: Optional[str] = Field(None, max_length=50)
    npi: str = Field(..., max_length=20)
    tax_id: Optional[str] = Field(None, max_length=20)
    specialty: Optional[str] = Field(None, max_length=100)
    
    # Address Information
    address_line1: Optional[str] = Field(None, max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=2)
    zip_code: Optional[str] = Field(None, max_length=10)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=255)
    fax: Optional[str] = Field(None, max_length=20)
    
    # Professional Information
    taxonomy_code: Optional[str] = Field(None, max_length=20)
    license_number: Optional[str] = Field(None, max_length=50)
    license_state: Optional[str] = Field(None, max_length=2)
    location: Optional[str] = Field(None, max_length=255)

class ProviderCreate(ProviderBase):
    pass

class ProviderUpdate(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    middle_initial: Optional[str] = Field(None, max_length=10)
    credentials: Optional[str] = Field(None, max_length=50)
    npi: Optional[str] = Field(None, max_length=20)
    tax_id: Optional[str] = Field(None, max_length=20)
    specialty: Optional[str] = Field(None, max_length=100)
    address_line1: Optional[str] = Field(None, max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=2)
    zip_code: Optional[str] = Field(None, max_length=10)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=255)
    fax: Optional[str] = Field(None, max_length=20)
    location: Optional[str] = Field(None, max_length=255)

class ProviderOut(ProviderBase, BaseModelWithTimestamps):
    provider_id: UUID
    
    class Config:
        from_attributes = True

# Insurance Models
class PatientInsuranceBase(BaseModel):
    patient_id: UUID
    insurance_type: InsuranceTypeEnum
    policy_number: Optional[str] = Field(None, max_length=50)
    group_number: Optional[str] = Field(None, max_length=50)
    plan_name: Optional[str] = Field(None, max_length=255)
    
    # Insured Information
    insured_name: Optional[str] = Field(None, max_length=255)
    insured_dob: Optional[date] = None
    insured_sex: Optional[GenderEnum] = None
    relationship_to_patient: Optional[RelationshipEnum] = None
    
    # Insured Address
    address_line1: Optional[str] = Field(None, max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=2)
    zip_code: Optional[str] = Field(None, max_length=10)
    phone: Optional[str] = Field(None, max_length=20)
    
    # Secondary Insurance
    other_insured_name: Optional[str] = Field(None, max_length=255)
    other_insured_policy_number: Optional[str] = Field(None, max_length=50)
    other_insured_group_number: Optional[str] = Field(None, max_length=50)
    other_insured_dob: Optional[date] = None
    other_insured_sex: Optional[GenderEnum] = None
    other_insured_employer: Optional[str] = Field(None, max_length=255)
    other_insured_plan_name: Optional[str] = Field(None, max_length=255)
    
    # Condition Related
    condition_employment_related: bool = False
    condition_auto_accident: bool = False
    condition_other_accident: bool = False
    
    # Coverage Dates
    effective_date: Optional[date] = None
    termination_date: Optional[date] = None

class PatientInsuranceCreate(PatientInsuranceBase):
    pass

class PatientInsuranceUpdate(BaseModel):
    insurance_type: Optional[InsuranceTypeEnum] = None
    policy_number: Optional[str] = Field(None, max_length=50)
    group_number: Optional[str] = Field(None, max_length=50)
    plan_name: Optional[str] = Field(None, max_length=255)
    insured_name: Optional[str] = Field(None, max_length=255)
    insured_dob: Optional[date] = None
    insured_sex: Optional[GenderEnum] = None
    relationship_to_patient: Optional[RelationshipEnum] = None
    effective_date: Optional[date] = None
    termination_date: Optional[date] = None

class PatientInsuranceOut(PatientInsuranceBase, BaseModelWithTimestamps):
    insurance_id: UUID
    
    class Config:
        from_attributes = True

# Service Line Models with AI Support
class ClaimServiceLineBase(BaseModel):
    line_number: Optional[int] = None
    service_date_from: date
    service_date_to: Optional[date] = None
    place_of_service_code: str = Field(..., max_length=2)
    emergency_indicator: Optional[str] = Field(None, max_length=1)
    cpt_hcpcs_code: str = Field(..., max_length=10)
    code_type: CodeTypeEnum = CodeTypeEnum.CPT  # New field
    modifier1: Optional[str] = Field(None, max_length=2)
    modifier2: Optional[str] = Field(None, max_length=2)
    modifier3: Optional[str] = Field(None, max_length=2)
    modifier4: Optional[str] = Field(None, max_length=2)
    diagnosis_pointer: Optional[str] = Field(None, max_length=12)
    charge_amount: Decimal = Field(..., decimal_places=2)
    units: Optional[int] = Field(1, ge=1)
    epsdt_indicator: Optional[str] = Field(None, max_length=1)
    rendering_provider_npi: Optional[str] = Field(None, max_length=20)
    rendering_provider_other_id: Optional[str] = Field(None, max_length=50)
    rendering_provider_qualifier: Optional[str] = Field(None, max_length=2)
    revenue_code: Optional[str] = Field(None, max_length=10)

class ClaimServiceLineCreate(ClaimServiceLineBase):
    # Allow optional CPT code for AI-assisted creation
    cpt_hcpcs_code: Optional[str] = Field(None, max_length=10)

class ClaimServiceLineUpdate(BaseModel):
    service_date_from: Optional[date] = None
    service_date_to: Optional[date] = None
    place_of_service_code: Optional[str] = Field(None, max_length=2)
    emergency_indicator: Optional[str] = Field(None, max_length=1)
    cpt_hcpcs_code: Optional[str] = Field(None, max_length=10)
    modifier1: Optional[str] = Field(None, max_length=2)
    modifier2: Optional[str] = Field(None, max_length=2)
    modifier3: Optional[str] = Field(None, max_length=2)
    modifier4: Optional[str] = Field(None, max_length=2)
    diagnosis_pointer: Optional[str] = Field(None, max_length=12)
    charge_amount: Optional[Decimal] = Field(None, decimal_places=2)
    units: Optional[int] = Field(None, ge=1)

class ClaimServiceLineOut(ClaimServiceLineBase, BaseModelWithTimestamps):
    service_line_id: UUID
    claim_id: UUID
    
    # Optional AI suggestions (not stored in DB, just for response)
    ai_suggestions: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

# Diagnosis Models
class ClaimDiagnosisBase(BaseModel):
    position: int = Field(..., ge=1, le=12)
    icd10_code: str = Field(..., max_length=10)
    diagnosis_pointer: Optional[str] = Field(None, pattern=r'^[A-L]$')

class ClaimDiagnosisCreate(ClaimDiagnosisBase):
    pass

class ClaimDiagnosisUpdate(BaseModel):
    position: Optional[int] = Field(None, ge=1, le=12)
    icd10_code: Optional[str] = Field(None, max_length=10)
    diagnosis_pointer: Optional[str] = Field(None, pattern=r'^[A-L]$')

class ClaimDiagnosisOut(ClaimDiagnosisBase, BaseModelWithTimestamps):
    claim_diagnosis_id: UUID
    claim_id: UUID
    
    class Config:
        from_attributes = True

# Document Models
class ClaimDocumentBase(BaseModel):
    document_type: str = Field(..., max_length=50)
    file_name: str = Field(..., max_length=255)
    content_type: str = Field(..., max_length=100)
    description: Optional[str] = None

class ClaimDocumentCreate(ClaimDocumentBase):
    file_data: bytes

class ClaimDocumentUpdate(BaseModel):
    document_type: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None

class ClaimDocumentResponse(ClaimDocumentBase, BaseModelWithTimestamps):
    document_id: UUID
    claim_id: UUID
    #file_data: Optional[bytes] = None
    ocr_redacted_text: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Main Claim Models
class ClaimBase(BaseModel):
    patient_id: UUID
    provider_id: UUID
    insurance_id: Optional[UUID] = None
    
    # Claim Identification
    claim_number: Optional[str] = Field(None, max_length=50)
    claim_date: date
    claim_status: ClaimStatusEnum = ClaimStatusEnum.DRAFT
    claim_type: Optional[str] = Field(None, max_length=50)
    submission_method: Optional[str] = Field(None, max_length=50)
    
    # Financial Information
    claim_amount: Optional[Decimal] = Field(None, decimal_places=2)
    total_charge: Optional[Decimal] = Field(None, decimal_places=2)
    amount_paid: Optional[Decimal] = Field(None, decimal_places=2)
    balance_due: Optional[Decimal] = Field(None, decimal_places=2)
    
    # Box 10 - Condition Related
    condition_employment_related: bool = False
    condition_auto_accident: bool = False
    condition_auto_accident_state: Optional[str] = Field(None, max_length=2)
    condition_other_accident: bool = False
    condition_claim_codes: Optional[str] = Field(None, max_length=100)
    
    # Box 14-19 - Additional Dates and References
    date_current_illness: Optional[date] = None
    date_first_consultation: Optional[date] = None
    date_similar_illness: Optional[date] = None
    dates_unable_to_work_from: Optional[date] = None
    dates_unable_to_work_to: Optional[date] = None
    referring_provider_name: Optional[str] = Field(None, max_length=255)
    referring_provider_npi: Optional[str] = Field(None, max_length=20)
    referring_provider_other_id: Optional[str] = Field(None, max_length=50)
    hospitalization_date_from: Optional[date] = None
    hospitalization_date_to: Optional[date] = None
    additional_claim_info: Optional[str] = None
    
    # Box 20 - Outside Lab
    outside_lab: bool = False
    outside_lab_charges: Optional[Decimal] = Field(None, decimal_places=2)
    
    # Box 22 - Resubmission
    resubmission_code: Optional[str] = Field(None, max_length=10)
    original_reference_number: Optional[str] = Field(None, max_length=50)
    
    # Box 23 - Authorization
    authorization_number: Optional[str] = Field(None, max_length=50)
    
    # Box 25-27 - Provider Information
    tax_id_type: Optional[TaxIdTypeEnum] = None
    patient_account_number: Optional[str] = Field(None, max_length=50)
    accept_assignment: Optional[bool] = None
    
    # Box 31 - Signatures
    physician_signature_date: Optional[date] = None
    physician_signature_on_file: bool = False
    
    # Box 32 - Service Facility
    service_facility_name: Optional[str] = Field(None, max_length=255)
    service_facility_address_line1: Optional[str] = Field(None, max_length=255)
    service_facility_address_line2: Optional[str] = Field(None, max_length=255)
    service_facility_city: Optional[str] = Field(None, max_length=100)
    service_facility_state: Optional[str] = Field(None, max_length=2)
    service_facility_zip_code: Optional[str] = Field(None, max_length=10)
    service_facility_npi: Optional[str] = Field(None, max_length=20)
    service_facility_other_id: Optional[str] = Field(None, max_length=50)
    
    # Box 33 - Billing Provider
    billing_provider_name: Optional[str] = Field(None, max_length=255)
    billing_provider_address_line1: Optional[str] = Field(None, max_length=255)
    billing_provider_address_line2: Optional[str] = Field(None, max_length=255)
    billing_provider_city: Optional[str] = Field(None, max_length=100)
    billing_provider_state: Optional[str] = Field(None, max_length=2)
    billing_provider_zip_code: Optional[str] = Field(None, max_length=10)
    billing_provider_phone: Optional[str] = Field(None, max_length=20)
    billing_provider_npi: Optional[str] = Field(None, max_length=20)
    billing_provider_other_id: Optional[str] = Field(None, max_length=50)
    
    # Additional Tracking
    submission_date: Optional[date] = None
    received_date: Optional[date] = None
    processed_date: Optional[date] = None
    paid_date: Optional[date] = None
    claim_frequency_code: Optional[str] = Field("1", max_length=2)
    medical_record_number: Optional[str] = Field(None, max_length=50)
    demonstration_project_id: Optional[str] = Field(None, max_length=50)
    
    # Legacy/Analysis Fields (keeping for backward compatibility)
    primary_diagnosis_code: Optional[str] = Field(None, max_length=10)
    primary_procedure_code: Optional[str] = Field(None, max_length=10)
    predicted_payout: Optional[Decimal] = Field(None, decimal_places=2)
    approval_probability: Optional[Decimal] = Field(None, ge=0, le=1)
    fraud_flag: bool = False
    fraud_reason: Optional[str] = None
    coverage_notes: Optional[str] = None
    suggested_diagnosis_code: Optional[str] = Field(None, max_length=10)
    suggested_procedure_code: Optional[str] = Field(None, max_length=10)

class ClaimCreate(ClaimBase):
    service_lines: List[ClaimServiceLineCreate] = Field(default_factory=list)
    diagnoses: List[ClaimDiagnosisCreate] = Field(default_factory=list)

class ClaimUpdate(BaseModel):
    claim_status: Optional[ClaimStatusEnum] = None
    claim_amount: Optional[Decimal] = None
    total_charge: Optional[Decimal] = None
    amount_paid: Optional[Decimal] = None
    balance_due: Optional[Decimal] = None
    authorization_number: Optional[str] = None
    coverage_notes: Optional[str] = None
    additional_claim_info: Optional[str] = None
    outside_lab: Optional[bool] = None
    outside_lab_charges: Optional[Decimal] = None
    physician_signature_date: Optional[date] = None
    physician_signature_on_file: Optional[bool] = None

class ClaimOut(ClaimBase, BaseModelWithTimestamps):
    claim_id: UUID
    
    # Related data (optional includes)
    service_lines: Optional[List[ClaimServiceLineOut]] = None
    diagnoses: Optional[List[ClaimDiagnosisOut]] = None
    patient: Optional[PatientOut] = None
    provider: Optional[ProviderOut] = None
    insurance: Optional[PatientInsuranceOut] = None
    
    # Optional AI suggestions (not stored in DB, just for response)
    ai_suggestions: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

# Response Models
class ClaimListResponse(BaseModel):
    claims: List[ClaimOut]
    total_count: int
    page: int
    page_size: int

class ClaimSummaryResponse(BaseModel):
    total_claims: int
    total_amount: Decimal
    pending_claims: int
    approved_claims: int
    denied_claims: int

# AI-specific models
class CodeSuggestion(BaseModel):
    code: str
    description: str
    confidence: float
    similarity: Optional[float] = None
    code_type: Optional[str] = None  # CPT, HCPCS, or ICD10

class ServiceLineWithSuggestions(BaseModel):
    service_line: ClaimServiceLineOut
    ai_suggestions: Optional[Dict[str, Any]] = None
    suggestion_applied: bool = False

class CodeSuggestionRequest(BaseModel):
    notes: str
    diagnosis_codes: Optional[List[str]] = []
    coverage_notes: Optional[str] = ""
    top_n: int = 3

class CodeSuggestion(BaseModel):
    code: str
    description: str
    confidence: float
    similarity: Optional[float] = None

class CodeSuggestionResponse(BaseModel):
    icd_suggestions: List[CodeSuggestion]
    procedure_suggestions: List[CodeSuggestion]  # Will include both CPT and HCPCS
    search_method: str
    notes_analyzed: str
    best_match_type: Optional[str] = None  # Indicates if best match is CPT or HCPCS

class HCPCSCodeSuggestion(BaseModel):
    code: str
    description: str
    confidence: float
    similarity: Optional[float] = None
    category: Optional[str] = None  # HCPCS category (A, B, C, etc.)

class EnhancedCodeSuggestionResponse(BaseModel):
    icd_suggestions: List[CodeSuggestion]
    cpt_suggestions: List[CodeSuggestion]
    hcpcs_suggestions: List[HCPCSCodeSuggestion]
    best_procedure_match: CodeSuggestion  # Single best match from CPT or HCPCS
    search_method: str
    notes_analyzed: str


class ServiceLineSuggestionRequest(BaseModel):
    service_line_data: Dict[str, Any]
    claim_diagnoses: List[str] = []
    coverage_notes: Optional[str] = ""

class ServiceLineSuggestionResponse(BaseModel):
    suggested_cpt_codes: List[CodeSuggestion]
    suggested_diagnosis_pointer: str
    confidence_score: float
    reasoning: str

# Segment Models
class SegmentBase(BaseModel):
    name: str = Field(..., max_length=255)
    description: Optional[str] = None

class SegmentCreate(SegmentBase):
    pass

class SegmentUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None

class SegmentOut(SegmentBase, BaseModelWithTimestamps):
    segment_id: UUID
    
    class Config:
        from_attributes = True

# Legacy compatibility - keeping original names for backward compatibility
class PatientCreate_Legacy(BaseModel):
    """Legacy patient create model for backward compatibility"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    patient_income: Optional[float] = None
    patient_marital_status: Optional[str] = None
    patient_employment_status: Optional[str] = None

# For backward compatibility with existing frontend
ClaimCreate_Legacy = ClaimCreate  # Alias for existing code


# User Role Enum
class UserRoleEnum(str, Enum):
    ADMIN = "ADMIN"
    MEDICAL_STAFF = "MEDICAL_STAFF"
    INSURANCE_STAFF = "INSURANCE_STAFF"  
    BILLING_STAFF = "BILLING_STAFF"

# Authentication Models
class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[UUID] = None
    role: Optional[UserRoleEnum] = None

# User Models
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    first_name: str = Field(..., max_length=100)
    last_name: str = Field(..., max_length=100)
    middle_initial: Optional[str] = Field(None, max_length=10)
    role: UserRoleEnum
    
    # Contact Information
    phone: Optional[str] = Field(None, max_length=20)
    address_line1: Optional[str] = Field(None, max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=2)
    zip_code: Optional[str] = Field(None, max_length=10)
    
    # Professional Information (for medical staff)
    npi: Optional[str] = Field(None, max_length=20)
    license_number: Optional[str] = Field(None, max_length=50)
    specialty: Optional[str] = Field(None, max_length=100)
    organization: Optional[str] = Field(None, max_length=255)
    department: Optional[str] = Field(None, max_length=100)
    
    # Insurance Staff Information
    insurance_company: Optional[str] = Field(None, max_length=255)
    
    # Status
    is_active: bool = True
    is_verified: bool = False

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    middle_initial: Optional[str] = Field(None, max_length=10)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=20)
    address_line1: Optional[str] = Field(None, max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=2)
    zip_code: Optional[str] = Field(None, max_length=10)
    specialty: Optional[str] = Field(None, max_length=100)
    organization: Optional[str] = Field(None, max_length=255)
    department: Optional[str] = Field(None, max_length=100)
    insurance_company: Optional[str] = Field(None, max_length=255)

class UserOut(UserBase, BaseModelWithTimestamps):
    user_id: UUID
    last_login: Optional[datetime] = None
    permissions: Optional[List[str]] = []
    
    class Config:
        from_attributes = True

class CurrentUser(BaseModel):
    user_id: UUID
    username: str
    email: str
    first_name: str
    last_name: str
    role: UserRoleEnum
    permissions: List[str]
    is_active: bool
    organization: Optional[str] = None

# Password Management
class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

# Permission Models
class PermissionBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    resource: str = Field(..., max_length=50)
    action: str = Field(..., max_length=50)

class PermissionCreate(PermissionBase):
    pass

class PermissionOut(PermissionBase, BaseModelWithTimestamps):
    permission_id: UUID
    
    class Config:
        from_attributes = True

class UserPermissionCreate(BaseModel):
    user_id: UUID
    permission_id: UUID
    resource_id: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None

class UserPermissionOut(BaseModel):
    user_permission_id: UUID
    user_id: UUID
    permission_id: UUID
    resource_id: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    granted_at: datetime
    granted_by: Optional[UUID] = None
    expires_at: Optional[datetime] = None
    permission: PermissionOut
    
    class Config:
        from_attributes = True

# Audit Log Models
class AuditLogCreate(BaseModel):
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    description: Optional[str] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    status: str = "SUCCESS"
    error_message: Optional[str] = None

class AuditLogOut(BaseModel):
    audit_id: UUID
    user_id: Optional[UUID] = None
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    description: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    user: Optional[UserOut] = None
    
    class Config:
        from_attributes = True

# Role-based Permissions Configuration
ROLE_PERMISSIONS = {
    "ADMIN": [
        # Full access to everything
        "users:create", "users:read", "users:update", "users:delete", "users:list",
        "patients:create", "patients:read", "patients:update", "patients:delete", "patients:list",
        "providers:create", "providers:read", "providers:update", "providers:delete", "providers:list",
        "claims:create", "claims:read", "claims:update", "claims:delete", "claims:list",
        "insurances:create", "insurances:read", "insurances:update", "insurances:delete", "insurances:list",
        "documents:create", "documents:read", "documents:update", "documents:delete", "documents:list",
        "ai:use", "reports:view", "audit:view", "settings:manage"
    ],
    
    "MEDICAL_STAFF": [
        # Can create and manage claims, patients, and use AI
        "patients:create", "patients:read", "patients:update", "patients:list",
        "providers:read", "providers:list",
        "claims:create", "claims:read", "claims:update", "claims:list",
        "insurances:read", "insurances:list",
        "documents:create", "documents:read", "documents:update", "documents:list",
        "ai:use", "reports:view"
    ],
    
    "INSURANCE_STAFF": [
        # Can view and process claims, limited patient access
        "patients:read", "patients:list",
        "providers:read", "providers:list", 
        "claims:read", "claims:update", "claims:list",
        "insurances:create", "insurances:read", "insurances:update", "insurances:list",
        "documents:read", "documents:list",
        "reports:view"
    ],
    
    "BILLING_STAFF": [
        # Can handle billing aspects of claims
        "patients:read", "patients:list",
        "providers:read", "providers:list",
        "claims:read", "claims:update", "claims:list",
        "insurances:read", "insurances:list",
        "documents:read", "documents:list",
        "reports:view"
    ]
}

# Dashboard Configuration by Role
ROLE_DASHBOARD_CONFIG = {
    "ADMIN": {
        "sections": ["overview", "users", "claims", "patients", "providers", "insurances", "reports", "audit"],
        "widgets": ["total_users", "active_claims", "recent_activity", "system_health"],
        "permissions": ["full_access"]
    },
    
    "MEDICAL_STAFF": {
        "sections": ["dashboard", "patients", "claims", "providers", "ai_suggestions"],
        "widgets": ["my_claims", "pending_claims", "recent_patients", "ai_suggestions"],
        "permissions": ["create_claims", "manage_patients", "use_ai"]
    },
    
    "INSURANCE_STAFF": {
        "sections": ["dashboard", "claims_review", "insurances", "reports"],
        "widgets": ["pending_reviews", "approved_claims", "denied_claims", "processing_time"],
        "permissions": ["review_claims", "manage_insurance"]
    },
    
    "BILLING_STAFF": {
        "sections": ["dashboard", "billing", "claims_status", "reports"],
        "widgets": ["billing_summary", "payment_status", "outstanding_claims", "collections"],
        "permissions": ["manage_billing", "view_financial"]
    }
}