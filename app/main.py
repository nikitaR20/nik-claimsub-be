from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import claim, provider, segment, claim_documents, patient,ai
from app import models, database

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(provider.router, tags=["Providers"])
app.include_router(segment.router, tags=["Segments"])
app.include_router(claim.router, tags=["Claims"])
app.include_router(claim_documents.router, tags=["Claim Documents"])
app.include_router(patient.router, tags=["Patients"])
app.include_router(ai.router)
