from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime

class ProviderOut(BaseModel):
    provider_id: UUID
    first_name: str
    last_name: str
    location: Optional[str]

    class Config:
        orm_mode = True


class SegmentOut(BaseModel):
    segment_id: UUID
    name: str
    description: Optional[str]

    class Config:
        orm_mode = True


class ClaimBase(BaseModel):
    provider_id: UUID
    risk_id: Optional[UUID] = None
    status: Optional[str] = "To Do"
    submission_date: Optional[date] = None
    summary: Optional[str] = None
    ex_gratia_flag: Optional[bool] = False
    appeal_case_flag: Optional[bool] = False
    reason_code: Optional[str] = None
    reason_description: Optional[str] = None
    last_status_update_date: Optional[datetime] = None


class ClaimCreate(ClaimBase):
    pass


class ClaimOut(ClaimBase):
    claim_id: UUID

    class Config:
        orm_mode = True
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime

class ClaimDocumentCreate(BaseModel):
    claim_id: UUID
    document_type: str = Field(..., description="Type of document uploaded")
    description: Optional[str] = None

class ClaimDocumentResponse(BaseModel):
    document_id: UUID
    claim_id: UUID
    document_type: str
    file_name: str
    content_type: str
    description: Optional[str]
    uploaded_at: datetime

    class Config:
        orm_mode = True