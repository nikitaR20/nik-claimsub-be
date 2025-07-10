from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import models, database
from app.routers import claim, provider, segment, claim_documents

# Create database tables
models.Base.metadata.create_all(bind=database.engine)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(provider.router)
app.include_router(segment.router)
app.include_router(claim.router)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Claims Backend API"}

app.include_router(claim_documents.router)
