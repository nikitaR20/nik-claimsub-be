# app/routers/ai.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ai.semantic_search import semantic_search

router = APIRouter(prefix="/ai", tags=["AI"])

class CoverageNotesPayload(BaseModel):
    coverage_notes: str

@router.post("/suggest_codes", response_model=dict)
def suggest_codes(payload: CoverageNotesPayload):
    notes = payload.coverage_notes.strip()
    if not notes:
        raise HTTPException(status_code=400, detail="Coverage notes are required")

    icd10_suggestions, proc_suggestions = semantic_search(notes, top_n=3)

    return {
        "suggested_diagnosis_codes": icd10_suggestions,
        "suggested_procedure_codes": proc_suggestions
    }
