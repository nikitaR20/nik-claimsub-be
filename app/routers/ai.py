# app/routers/ai.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ai.semantic_search import semantic_search

router = APIRouter(prefix="/ai", tags=["AI"])

# Initialize semantic search on module load
#initialize_search()

class CoverageNotesPayload(BaseModel):
    coverage_notes: str

@router.post("/suggest_codes", response_model=dict)
def suggest_codes(payload: CoverageNotesPayload):
    notes = payload.coverage_notes.strip()
    if not notes:
        raise HTTPException(status_code=400, detail="Coverage notes are required")

    # Returns lists of dicts for ICD-10 and procedure codes
    icd10_suggestions, proc_suggestions = semantic_search(notes, top_n=3)

    return {
        "suggested_diagnosis_codes": icd10_suggestions,      # list of {code, description}
        "suggested_procedure_codes": proc_suggestions       # list of {code, description}
    }
