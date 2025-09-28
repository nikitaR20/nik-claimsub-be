from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app import schemas
from app.database import get_db
from app.ai.semantic_search import enhanced_semantic_search  # Updated import
import logging
import os

router = APIRouter(prefix="/ai", tags=["AI Suggestions"])

SEARCH_MODE = os.getenv("SEARCH_MODE", "db")

@router.post("/suggest-codes", response_model=schemas.EnhancedCodeSuggestionResponse)
def suggest_codes(
    request: schemas.CodeSuggestionRequest,
    db: Session = Depends(get_db)
):
    """Get AI-powered ICD-10, CPT, and HCPCS code suggestions based on clinical notes"""
    try:
        notes_lower = request.notes.lower()
        
        # --- AI/Embedding Mode with HCPCS support ---
        if SEARCH_MODE == "ai":
            icd_results, cpt_results, hcpcs_results = enhanced_semantic_search(
                request.notes, top_n=request.top_n
            )
            
            icd_suggestions = [
                schemas.CodeSuggestion(
                    code=r["code"],
                    description=r["description"],
                    confidence=r.get("similarity", 0.5),
                    code_type="ICD10"
                )
                for r in icd_results
            ]
            
            cpt_suggestions = [
                schemas.CodeSuggestion(
                    code=r["code"],
                    description=r["description"],
                    confidence=r.get("similarity", 0.5),
                    code_type="CPT"
                )
                for r in cpt_results
            ]
            
            hcpcs_suggestions = [
                schemas.HCPCSCodeSuggestion(
                    code=r["code"],
                    description=r["description"],
                    confidence=r.get("similarity", 0.5),
                    category=r["code"][0] if r["code"] else None  # First letter indicates category
                )
                for r in hcpcs_results
            ]
            
            # Determine best procedure match (CPT vs HCPCS)
            best_procedure_match = None
            if cpt_results and hcpcs_results:
                best_cpt = cpt_results[0] if cpt_results else None
                best_hcpcs = hcpcs_results[0] if hcpcs_results else None
                
                if best_cpt and best_hcpcs:
                    if best_cpt.get("similarity", 0) >= best_hcpcs.get("similarity", 0):
                        best_procedure_match = schemas.CodeSuggestion(
                            code=best_cpt["code"],
                            description=best_cpt["description"],
                            confidence=best_cpt.get("similarity", 0.5),
                            code_type="CPT"
                        )
                    else:
                        best_procedure_match = schemas.CodeSuggestion(
                            code=best_hcpcs["code"],
                            description=best_hcpcs["description"],
                            confidence=best_hcpcs.get("similarity", 0.5),
                            code_type="HCPCS"
                        )
                elif best_cpt:
                    best_procedure_match = schemas.CodeSuggestion(
                        code=best_cpt["code"],
                        description=best_cpt["description"],
                        confidence=best_cpt.get("similarity", 0.5),
                        code_type="CPT"
                    )
                elif best_hcpcs:
                    best_procedure_match = schemas.CodeSuggestion(
                        code=best_hcpcs["code"],
                        description=best_hcpcs["description"],
                        confidence=best_hcpcs.get("similarity", 0.5),
                        code_type="HCPCS"
                    )
            
            # Fallback if no best match found
            if not best_procedure_match and (cpt_suggestions or hcpcs_suggestions):
                if cpt_suggestions:
                    best_procedure_match = cpt_suggestions[0]
                elif hcpcs_suggestions:
                    best_procedure_match = schemas.CodeSuggestion(
                        code=hcpcs_suggestions[0].code,
                        description=hcpcs_suggestions[0].description,
                        confidence=hcpcs_suggestions[0].confidence,
                        code_type="HCPCS"
                    )
            
            return schemas.EnhancedCodeSuggestionResponse(
                icd_suggestions=icd_suggestions,
                cpt_suggestions=cpt_suggestions,
                hcpcs_suggestions=hcpcs_suggestions,
                best_procedure_match=best_procedure_match,
                search_method="enhanced_semantic_search",
                notes_analyzed=request.notes[:100] + "..." if len(request.notes) > 100 else request.notes
            )

        # --- Original keyword-based fallback with HCPCS ---
        icd_suggestions = []
        cpt_suggestions = []
        hcpcs_suggestions = []

        # Enhanced keyword matching for different code types
        conditions = {
            'hypertension': {'code': 'I10', 'description': 'Essential hypertension', 'confidence': 0.85, 'type': 'ICD10'},
            'high blood pressure': {'code': 'I10', 'description': 'Essential hypertension', 'confidence': 0.85, 'type': 'ICD10'},
            'diabetes': {'code': 'E11.9', 'description': 'Type 2 diabetes mellitus without complications', 'confidence': 0.80, 'type': 'ICD10'},
            'chest pain': {'code': 'R06.02', 'description': 'Shortness of breath', 'confidence': 0.75, 'type': 'ICD10'},
        }
        
        # CPT procedures
        cpt_procedures = {
            'office visit': {'code': '99213', 'description': 'Office visit, established patient, low complexity', 'confidence': 0.80, 'type': 'CPT'},
            'blood test': {'code': '80053', 'description': 'Comprehensive metabolic panel', 'confidence': 0.75, 'type': 'CPT'},
            'x-ray': {'code': '73060', 'description': 'Radiologic examination, knee; 1 or 2 views', 'confidence': 0.70, 'type': 'CPT'},
            'ekg': {'code': '93000', 'description': 'Electrocardiogram, routine ECG with at least 12 leads', 'confidence': 0.80, 'type': 'CPT'},
        }
        
        # HCPCS supplies/services
        hcpcs_items = {
            'wheelchair': {'code': 'E1130', 'description': 'Standard wheelchair', 'confidence': 0.85, 'type': 'HCPCS'},
            'oxygen': {'code': 'E0424', 'description': 'Stationary oxygen concentrator', 'confidence': 0.80, 'type': 'HCPCS'},
            'walker': {'code': 'E0130', 'description': 'Walker, rigid, wheeled', 'confidence': 0.85, 'type': 'HCPCS'},
            'blood glucose': {'code': 'A4253', 'description': 'Blood glucose test strips', 'confidence': 0.80, 'type': 'HCPCS'},
            'injection': {'code': 'J3420', 'description': 'Injection, vitamin B-12 cyanocobalamin', 'confidence': 0.70, 'type': 'HCPCS'},
            'ambulance': {'code': 'A0429', 'description': 'Ambulance service, basic life support', 'confidence': 0.90, 'type': 'HCPCS'},
        }
        
        # Check for matches
        for condition, details in conditions.items():
            if condition in notes_lower:
                icd_suggestions.append(schemas.CodeSuggestion(
                    code=details['code'],
                    description=details['description'],
                    confidence=details['confidence'],
                    code_type=details['type']
                ))
        
        for procedure, details in cpt_procedures.items():
            if procedure in notes_lower:
                cpt_suggestions.append(schemas.CodeSuggestion(
                    code=details['code'],
                    description=details['description'],
                    confidence=details['confidence'],
                    code_type=details['type']
                ))
        
        for item, details in hcpcs_items.items():
            if item in notes_lower:
                hcpcs_suggestions.append(schemas.HCPCSCodeSuggestion(
                    code=details['code'],
                    description=details['description'],
                    confidence=details['confidence'],
                    category=details['code'][0]
                ))
        
        # Remove duplicates and sort
        icd_suggestions = list({v.code: v for v in icd_suggestions}.values())
        cpt_suggestions = list({v.code: v for v in cpt_suggestions}.values())
        hcpcs_suggestions = list({v.code: v for v in hcpcs_suggestions}.values())
        
        icd_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        cpt_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        hcpcs_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Determine best procedure match
        all_procedure_suggestions = []
        for cpt in cpt_suggestions:
            all_procedure_suggestions.append(cpt)
        for hcpcs in hcpcs_suggestions:
            all_procedure_suggestions.append(schemas.CodeSuggestion(
                code=hcpcs.code,
                description=hcpcs.description,
                confidence=hcpcs.confidence,
                code_type="HCPCS"
            ))
        
        best_procedure_match = None
        if all_procedure_suggestions:
            best_procedure_match = max(all_procedure_suggestions, key=lambda x: x.confidence)
        
        # Provide defaults if nothing found
        if not icd_suggestions:
            icd_suggestions.append(schemas.CodeSuggestion(
                code='Z00.00',
                description='Encounter for general adult medical examination without abnormal findings',
                confidence=0.30,
                code_type='ICD10'
            ))
            
        if not cpt_suggestions and not hcpcs_suggestions:
            cpt_suggestions.append(schemas.CodeSuggestion(
                code='99213',
                description='Office visit, established patient, low complexity',
                confidence=0.30,
                code_type='CPT'
            ))
            best_procedure_match = cpt_suggestions[0]

        return schemas.EnhancedCodeSuggestionResponse(
            icd_suggestions=icd_suggestions[:request.top_n],
            cpt_suggestions=cpt_suggestions[:request.top_n],
            hcpcs_suggestions=hcpcs_suggestions[:request.top_n],
            best_procedure_match=best_procedure_match,
            search_method="enhanced_keyword_matching",
            notes_analyzed=request.notes[:100] + "..." if len(request.notes) > 100 else request.notes
        )
        
    except Exception as e:
        logging.error(f"Error in AI code suggestion: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate AI suggestions")

# Backward compatibility endpoint
@router.post("/suggest-codes-legacy", response_model=schemas.CodeSuggestionResponse)
def suggest_codes_legacy(
    request: schemas.CodeSuggestionRequest,
    db: Session = Depends(get_db)
):
    """Legacy endpoint for backward compatibility"""
    try:
        # Get enhanced suggestions
        enhanced_response = suggest_codes(request, db)
        
        # Combine CPT and HCPCS into procedure suggestions
        procedure_suggestions = enhanced_response.cpt_suggestions + [
            schemas.CodeSuggestion(
                code=h.code,
                description=h.description,
                confidence=h.confidence,
                code_type="HCPCS"
            ) for h in enhanced_response.hcpcs_suggestions
        ]
        
        # Sort by confidence
        procedure_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return schemas.CodeSuggestionResponse(
            icd_suggestions=enhanced_response.icd_suggestions,
            procedure_suggestions=procedure_suggestions[:request.top_n],
            search_method=enhanced_response.search_method,
            notes_analyzed=enhanced_response.notes_analyzed
        )
        
    except Exception as e:
        logging.error(f"Error in legacy AI code suggestion: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate AI suggestions")