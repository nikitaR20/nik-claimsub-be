from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import claim, provider, segment, claim_documents, patient
from app import models, database

# Create database tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(provider.router, prefix="/providers", tags=["Providers"])
app.include_router(segment.router, prefix="/segments", tags=["Segments"])
app.include_router(claim.router, prefix="/claims", tags=["Claims"])
app.include_router(claim_documents.router, prefix="/claim-documents", tags=["Claim Documents"])
app.include_router(patient.router, prefix="/patients", tags=["Patients"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Claims Backend API"}
