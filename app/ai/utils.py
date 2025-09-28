# app/ai/utils.py
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def load_table(table_name):
    """Load a table from the database into a pandas DataFrame"""
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql_table(table_name, engine)
        return df
    except Exception as e:
        print(f"Error loading table {table_name}: {str(e)}")
        return pd.DataFrame()

def get_ai_status():
    """Check if AI services are available and configured"""
    try:
        from app.ai.semantic_search import semantic_search
        from app.ai.ai_embeddings import SEARCH_MODE
        return {
            "available": True,
            "search_mode": SEARCH_MODE,
            "features": ["code_suggestions", "semantic_search", "claim_analysis"]
        }
    except ImportError as e:
        return {
            "available": False,
            "error": str(e),
            "search_mode": None,
            "features": []
        }

def safe_ai_suggestion(context, top_n=3, fallback_codes=None):
    """
    Safely get AI suggestions with fallback to default codes
    """
    try:
        from app.ai.semantic_search import semantic_search
        icd_results, proc_results = semantic_search(context, top_n)
        return {
            "success": True,
            "icd_suggestions": icd_results,
            "procedure_suggestions": proc_results,
            "context": context
        }
    except Exception as e:
        # Return fallback codes if AI fails
        fallback_icd = fallback_codes.get('icd', []) if fallback_codes else []
        fallback_proc = fallback_codes.get('proc', []) if fallback_codes else []
        
        return {
            "success": False,
            "error": str(e),
            "icd_suggestions": fallback_icd,
            "procedure_suggestions": fallback_proc,
            "context": context,
            "fallback_used": True
        }

def format_diagnosis_pointer(diagnosis_count):
    """Format diagnosis pointers (A, B, C, etc.) based on count"""
    if diagnosis_count == 0:
        return ""
    
    pointers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    return ",".join(pointers[:min(diagnosis_count, 12)])

def extract_context_from_claim(claim):
    """Extract relevant context from a claim for AI analysis"""
    context_parts = []
    
    # Add coverage notes
    if hasattr(claim, 'coverage_notes') and claim.coverage_notes:
        context_parts.append(claim.coverage_notes)
    
    # Add additional claim info
    if hasattr(claim, 'additional_claim_info') and claim.additional_claim_info:
        context_parts.append(claim.additional_claim_info)
    
    # Add existing diagnoses for context
    if hasattr(claim, 'diagnoses') and claim.diagnoses:
        diag_codes = [d.icd10_code for d in claim.diagnoses]
        context_parts.append(f"Existing diagnoses: {', '.join(diag_codes)}")
    
    # Add existing procedures for context
    if hasattr(claim, 'service_lines') and claim.service_lines:
        cpt_codes = [sl.cpt_hcpcs_code for sl in claim.service_lines]
        context_parts.append(f"Existing procedures: {', '.join(cpt_codes)}")
    
    return " | ".join(context_parts) if context_parts else None

def validate_suggested_codes(icd_codes, proc_codes, db_session):
    """Validate that suggested codes exist in the database"""
    validated = {"icd": [], "proc": []}
    
    # Validate ICD codes
    for code in icd_codes:
        result = db_session.execute(
            "SELECT code, description FROM icd10_codes WHERE code = :code",
            {"code": code}
        ).fetchone()
        if result:
            validated["icd"].append({"code": result[0], "description": result[1]})
    
    # Validate procedure codes
    for code in proc_codes:
        result = db_session.execute(
            "SELECT code, description FROM procedural_codes WHERE code = :code",
            {"code": code}
        ).fetchone()
        if result:
            validated["proc"].append({"code": result[0], "description": result[1]})
    
    return validated

def get_code_description(code, code_type, db_session):
    """Get description for a specific code"""
    try:
        if code_type.lower() == 'icd':
            result = db_session.execute(
                "SELECT description FROM icd10_codes WHERE code = :code",
                {"code": code}
            ).fetchone()
        else:  # procedure/cpt
            result = db_session.execute(
                "SELECT description FROM procedural_codes WHERE code = :code",
                {"code": code}
            ).fetchone()
        
        return result[0] if result else None
    except Exception:
        return None

# Default fallback codes for common scenarios
DEFAULT_FALLBACK_CODES = {
    "general": {
        "icd": [
            {"code": "Z00.00", "description": "Encounter for general adult medical examination without abnormal findings"},
            {"code": "M79.3", "description": "Panniculitis, unspecified"}
        ],
        "proc": [
            {"code": "99213", "description": "Office or other outpatient visit, established patient"},
            {"code": "99214", "description": "Office or other outpatient visit, established patient"}
        ]
    },
    "surgical": {
        "icd": [
            {"code": "Z51.11", "description": "Encounter for antineoplastic chemotherapy"},
            {"code": "M79.3", "description": "Panniculitis, unspecified"}
        ],
        "proc": [
            {"code": "19120", "description": "Excision of cyst, fibroadenoma, or other benign or malignant tumor"},
            {"code": "27447", "description": "Arthroplasty, knee, condyle and plateau"}
        ]
    }
}