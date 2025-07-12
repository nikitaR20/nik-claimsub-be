import uuid
from sqlalchemy import Column, String, ForeignKey, LargeBinary, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from sqlalchemy import (
    Column, String, Text, Boolean, Date, TIMESTAMP, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.database import Base


class Provider(Base):
    __tablename__ = "providers"
    
    provider_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    location = Column(Text)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


class Segment(Base):
    __tablename__ = "segments"  # previously was 'customer_segments'
    
    segment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


class Claim(Base):
    __tablename__ = "claims"

    claim_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider_id = Column(UUID(as_uuid=True), ForeignKey("providers.provider_id"), nullable=False)
    risk_id = Column(UUID(as_uuid=True), nullable=True)  # FK can be added later: ForeignKey("risk_ratings.risk_id")
    
    status = Column(String(50), nullable=False, default="To Do")
    submission_date = Column(Date, nullable=False, server_default=func.current_date())
    summary = Column(Text)
    ex_gratia_flag = Column(Boolean, default=False)
    appeal_case_flag = Column(Boolean, default=False)
    reason_code = Column(String(100))
    reason_description = Column(Text)

    last_status_update_date = Column(TIMESTAMP(timezone=True), server_default=func.now())
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

#added claim_documents class
class ClaimDocument(Base):
    __tablename__ = "claim_documents"

    document_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False)
    document_type = Column(String, nullable=False)  # We’ll validate this via frontend dropdown
    file_name = Column(String, nullable=False)
    file_data = Column(LargeBinary, nullable=False)
    content_type = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    # New column to store redacted OCR text
    ocr_redacted_text = Column(Text, nullable=True)