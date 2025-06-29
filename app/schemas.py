from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime
from typing import Optional, List

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
