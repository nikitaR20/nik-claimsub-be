from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import claim, provider, segment, claim_documents, patient, insurance,auth,ai

from app import models, database
import uuid
from sqlalchemy import Column, String, Text, Boolean, Date, DateTime, Float, ForeignKey, TIMESTAMP, LargeBinary, MetaData, Table, Integer, Numeric, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base
from app.routers import users  # Add this import


# Try to import AI router, but don't fail if AI modules are missing
try:
    from app.routers import ai
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("AI modules not available. AI features will be disabled.")

# Create all database tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="Healthcare Claims Management System",
    description="Complete CMS 1500 compliant claims processing system with AI-powered code suggestions",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(patient.router, tags=["Patients"])
app.include_router(provider.router, tags=["Providers"]) 
app.include_router(insurance.router, tags=["Insurance"])
app.include_router(claim.router, tags=["Claims"])
app.include_router(claim_documents.router, tags=["Claim Documents"])
app.include_router(auth.router) 
app.include_router(segment.router, tags=["Segments"])
app.include_router(users.router)
# Include AI router if available
if AI_AVAILABLE:
    app.include_router(ai.router, tags=["AI Services"])

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "ai_enabled": AI_AVAILABLE,
        "features": [
            "CMS 1500 Compliant Claims",
            "Patient Management",
            "Provider Management", 
            "Insurance Management",
            "Document Management",
            "AI-Powered Code Suggestions" if AI_AVAILABLE else "AI Disabled"
        ]
    }

# Root endpoint
@app.get("/")
def read_root():
    """Welcome endpoint with API information"""
    features = {
        "cms_1500_compliance": True,
        "patient_management": True,
        "provider_management": True,
        "insurance_management": True,
        "document_management": True,
        "ai_integration": AI_AVAILABLE
    }
    
    return {
        "message": "Healthcare Claims Management System API",
        "version": "2.1.0",
        "documentation": "/docs",
        "health": "/health",
        "features": features,
        "ai_status": "enabled" if AI_AVAILABLE else "disabled"
    }

# API versioning endpoints
@app.get("/api/v1")
def api_v1_info():
    """API v1 information (legacy)"""
    return {
        "version": "1.0.0", 
        "status": "deprecated",
        "message": "Please use v2 endpoints for full CMS 1500 compliance"
    }

@app.get("/api/v2")  
def api_v2_info():
    """API v2 information (current)"""
    endpoints = {
        "patients": "/patients",
        "providers": "/providers",
        "insurances": "/insurances", 
        "claims": "/claims",
        "documents": "/claim-documents",
        "segments": "/segments"
    }
    
    if AI_AVAILABLE:
        endpoints["ai"] = "/ai"
    
    return {
        "version": "2.1.0",
        "status": "current", 
        "cms_1500_compliant": True,
        "ai_powered": AI_AVAILABLE,
        "endpoints": endpoints
    }

