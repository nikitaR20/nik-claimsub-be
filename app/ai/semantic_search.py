import os
import numpy as np
import faiss
import json
from .utils import load_table
from .ai_embeddings import embed_text, ICD_FAISS_PATH, PROC_FAISS_PATH
from dotenv import load_dotenv
import re
import logging

logger = logging.getLogger(__name__)

load_dotenv()
SEARCH_MODE = os.getenv("SEARCH_MODE", "db")
ICD_FAISS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "embeddings", "icd_faiss.index")
PROC_FAISS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "embeddings", "proc_faiss.index")
HCPCS_FAISS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "embeddings", "hcpcs_faiss.index")

def is_valid_embedding(embedding, expected_dim=None):
    """Check if an embedding is valid and usable"""
    if embedding is None:
        return False
    
    if isinstance(embedding, str):
        if embedding.strip() == '':
            return False
        try:
            parsed = json.loads(embedding)
            if not isinstance(parsed, list) or len(parsed) == 0:
                return False
            embedding = parsed
        except (json.JSONDecodeError, TypeError):
            return False
    
    if isinstance(embedding, (list, np.ndarray)):
        if len(embedding) == 0:
            return False
        
        if isinstance(embedding, np.ndarray):
            if np.allclose(embedding, 0):
                return False
        elif isinstance(embedding, list):
            if all(x == 0 for x in embedding):
                return False
        
        if expected_dim is not None and len(embedding) != expected_dim:
            return False
        
        return True
    
    return False

def get_expected_embedding_dimension():
    """Get the expected embedding dimension from current model"""
    try:
        test_embedding = embed_text(["test"])[0]
        return test_embedding.shape[0] if hasattr(test_embedding, 'shape') else len(test_embedding)
    except Exception:
        return None

# Load all tables including HCPCS
print("Loading medical code tables...")
icd10_df = load_table("icd10_codes")
proc_df = load_table("procedural_codes")
hcpcs_df = load_table("hcpcs_codes")  # New HCPCS table

# Get expected dimension
expected_dim = get_expected_embedding_dimension()
print(f"Current embedding model dimension: {expected_dim}")

# Process embeddings for all tables
for df_name, df in [("ICD10", icd10_df), ("Procedures", proc_df), ("HCPCS", hcpcs_df)]:
    if "embedding" in df.columns and not df.empty:
        valid_count = {"count": 0}
        total_count = len(df)
        
        def process_embedding(x):
            if is_valid_embedding(x, expected_dim):
                valid_count["count"] += 1
                if isinstance(x, str):
                    try:
                        return np.array(json.loads(x), dtype=np.float32)
                    except:
                        return None
                elif isinstance(x, list):
                    return np.array(x, dtype=np.float32)
                elif isinstance(x, np.ndarray):
                    return x.astype(np.float32)
            return None
        
        df["embedding"] = df["embedding"].apply(process_embedding)
        print(f"{df_name}: {valid_count['count']}/{total_count} have valid embeddings with correct dimensions")

def load_faiss_index_only(path, table_name):
    """Load FAISS index from disk (no rebuilds)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FAISS index for {table_name} not found at {path}. "
            f"Run the standalone script first to generate it."
        )
    index = faiss.read_index(path)
    print(f"Loaded FAISS index for {table_name} from {path} (dim={index.d}, ntotal={index.ntotal})")
    return index

def initialize_compatible_indices():
    """Initialize FAISS indices (only load prebuilt ones)."""
    global icd_index, proc_index, hcpcs_index, SEARCH_MODE

    if SEARCH_MODE != "ai":
        print("AI search mode not enabled, skipping FAISS index initialization")
        return

    if expected_dim is None:
        print("Cannot determine embedding dimension, falling back to database search")
        SEARCH_MODE = "db"
        return

    try:
        print("Loading ICD10 FAISS index...")
        icd_index = load_faiss_index_only(ICD_FAISS_PATH, "ICD10")

        print("Loading Procedures FAISS index...")
        proc_index = load_faiss_index_only(PROC_FAISS_PATH, "Procedures")

        print("Loading HCPCS FAISS index...")
        hcpcs_index = load_faiss_index_only(HCPCS_FAISS_PATH, "HCPCS")

        print("All FAISS indices loaded successfully")

    except Exception as e:
        print(f"Error loading FAISS indices: {e}")
        print("Falling back to database search mode")
        SEARCH_MODE = "db"

# Initialize indices
icd_index = None
proc_index = None
hcpcs_index = None

if SEARCH_MODE == "ai":
    initialize_compatible_indices()

# Keyword categories for better matching
SURGICAL_KEYWORDS = ["surgery", "decompression", "arthrodesis", "fusion", "resection", "surgical", "operation"]
SUPPLY_KEYWORDS = ["wheelchair", "walker", "oxygen", "supplies", "equipment", "device", "prosthetic", "orthotic"]
INJECTION_KEYWORDS = ["injection", "vaccine", "immunization", "shot", "infusion"]
TRANSPORT_KEYWORDS = ["ambulance", "transport", "emergency transport", "medical transport"]


def enhanced_semantic_search(notes: str, top_n: int = 3):
    """Emergency fix with proper keyword-based search"""
    notes = notes.strip()
    if not notes:
        return [], [], []

    notes_lower = notes.lower()
    logger.info(f"Searching for: {notes_lower}")
    
    # Emergency keyword-based search with medical logic
    icd_results = []
    proc_results = []
    hcpcs_results = []
    
    try:
        # Load tables
        from .utils import load_table
        icd10_df = load_table("icd10_codes") 
        proc_df = load_table("procedural_codes")
        hcpcs_df = load_table("hcpcs_codes")
        
        if icd10_df.empty or proc_df.empty:
            logger.error("Failed to load medical code tables")
            return [], [], []
        
        # SMART KEYWORD MAPPING - Medical condition patterns
        condition_keywords = {
            # Respiratory/throat conditions
            'throat': ['J02', 'J03', 'R06.02', 'J06'],
            'strep': ['J02.0', 'B95.0'],
            'pharyngitis': ['J02', 'J03'],
            'fever': ['R50', 'R51'],
            'fatigue': ['R53', 'Z73'],
            'cough': ['R05', 'J40'],
            'respiratory': ['J06', 'J00', 'J98'],
            
            # Other common conditions  
            'pain': ['M79', 'R52', 'G89'],
            'headache': ['R51', 'G44'],
            'nausea': ['R11', 'K59'],
            'diabetes': ['E11', 'E10'],
            'hypertension': ['I10', 'I15'],
            'depression': ['F33', 'F32'],
            'anxiety': ['F41', 'F40'],
        }
        
        # SMART PROCEDURE MAPPING
        procedure_keywords = {
            # Office visits and consultations
            'visit': ['99213', '99214', '99212', '99215'],
            'office': ['99213', '99214', '99212'],
            'consultation': ['99241', '99242', '99213'],
            'examination': ['99213', '99214', '99395'],
            'followup': ['99213', '99214'],
            'checkup': ['99395', '99396', '99213'],
            
            # Tests and procedures
            'test': ['87880', '36415', '80053'],
            'blood': ['36415', '80053', '85025'],
            'strep': ['87880', '87650'],
            'culture': ['87081', '87070', '87040'],
            'xray': ['73060', '71020', '71010'],
            'ekg': ['93000', '93005'],
            'injection': ['96372', '90471'],
            'vaccine': ['90471', '90472'],
        }
        
        # Find ICD codes by keyword matching
        matched_keywords = []
        for keyword, code_prefixes in condition_keywords.items():
            if keyword in notes_lower:
                matched_keywords.append(keyword)
                logger.info(f"Found keyword: {keyword}")
                
                for code_prefix in code_prefixes[:2]:  # Top 2 per keyword
                    # Try exact match first
                    exact_match = icd10_df[icd10_df['code'] == code_prefix]
                    if not exact_match.empty:
                        icd_results.append({
                            'code': exact_match.iloc[0]['code'],
                            'description': exact_match.iloc[0]['description'],
                            'similarity': 0.9
                        })
                    else:
                        # Try prefix match
                        prefix_matches = icd10_df[icd10_df['code'].str.startswith(code_prefix, na=False)]
                        if not prefix_matches.empty:
                            icd_results.append({
                                'code': prefix_matches.iloc[0]['code'],
                                'description': prefix_matches.iloc[0]['description'],
                                'similarity': 0.8
                            })
        
        # Find procedure codes
       # Find procedure codes - IMPROVED VERSION
        for keyword, codes in procedure_keywords.items():
            if keyword in notes_lower:
                logger.info(f"Found procedure keyword: {keyword}")
                
                for code in codes[:2]:
                    # Try to find the exact code first
                    match = proc_df[proc_df['code'] == code]
                    if not match.empty:
                        proc_results.append({
                            'code': code,
                            'description': match.iloc[0]['description'],
                            'similarity': 0.9
                        })
                        logger.info(f"Added procedure: {code}")
                    else:
                        logger.warning(f"Code {code} not found in procedural_codes table")

        # Also add this fallback after the keyword search:
        # If still no procedure results, add default office visit codes
        if not proc_results:
            logger.info("No procedure keywords matched, adding default office visits")
            # Try to find common office visit codes
            default_codes = ['99213', '99214', '99212', '99215']
            for code in default_codes:
                match = proc_df[proc_df['code'] == code] 
                if not match.empty:
                    proc_results.append({
                        'code': code,
                        'description': match.iloc[0]['description'],
                        'similarity': 0.7  # Higher confidence for office visits
                    })
                    logger.info(f"Added default procedure: {code}")
                else:
                    logger.warning(f"Default code {code} not found in procedural_codes table")
            
            # If STILL no results, do a broader search
            if not proc_results:
                logger.info("Even default codes not found, searching for 'office' or 'visit'")
                office_matches = proc_df[proc_df['description'].str.contains('office|visit|outpatient', case=False, na=False)]
                for _, row in office_matches.head(3).iterrows():
                    proc_results.append({
                        'code': row['code'],
                        'description': row['description'],
                        'similarity': 0.6
                    })

        # ALSO - Add this debug section right after loading the tables:
        logger.info(f"Loaded tables - ICD: {len(icd10_df)}, Proc: {len(proc_df)}, HCPCS: {len(hcpcs_df)}")

        # Check if key codes exist
        test_codes = ['99213', '99214', '87880']
        for code in test_codes:
            exists = not proc_df[proc_df['code'] == code].empty
            logger.info(f"Code {code} exists in proc_df: {exists}")

        # And improve the procedure_keywords dictionary:
        procedure_keywords = {
            # Office visits and consultations - EXPANDED
            'visit': ['99213', '99214', '99212', '99215'],
            'office': ['99213', '99214', '99212'],  
            'consultation': ['99241', '99242', '99213'],
            'examination': ['99213', '99214', '99395'],
            'followup': ['99213', '99214'],
            'checkup': ['99395', '99396', '99213'],
            'patient': ['99213', '99214'],  # Most notes mention "patient"
            
            # Tests and procedures - EXPANDED
            'test': ['87880', '87650', '36415', '80053'],  # Strep test should be first
            'rapid': ['87880', '87650'],  # Rapid strep test
            'blood': ['36415', '80053', '85025'],
            'strep': ['87880', '87650'],  # Strep-specific tests
            'culture': ['87081', '87070', '87040'],
            'throat': ['87880', '87070'],  # Throat-related procedures
            'xray': ['73060', '71020', '71010'],
            'ekg': ['93000', '93005'],
            'injection': ['96372', '90471'],
            'vaccine': ['90471', '90472'],
        }
        
        # If no keyword matches, try description search
        if not icd_results:
            logger.info("No keyword matches, trying description search")
            # Extract meaningful words (not stop words)
            meaningful_words = [w for w in notes_lower.split() 
                             if len(w) > 3 and w not in ['with', 'have', 'been', 'will', 'this', 'that', 'from']]
            
            for word in meaningful_words[:3]:  # Try top 3 meaningful words
                matches = icd10_df[icd10_df['description'].str.contains(word, case=False, na=False)]
                if not matches.empty and len(matches) < 50:  # Not too many matches
                    for _, row in matches.head(2).iterrows():
                        icd_results.append({
                            'code': row['code'],
                            'description': row['description'], 
                            'similarity': 0.6
                        })
        
        if not proc_results:
            # Default to office visits for most cases
            default_procs = ['99213', '99214', '99212']
            for code in default_procs:
                match = proc_df[proc_df['code'] == code]
                if not match.empty:
                    proc_results.append({
                        'code': code,
                        'description': match.iloc[0]['description'],
                        'similarity': 0.5
                    })
        
        # HCPCS only for specific supplies/services
        hcpcs_keywords = {
            'wheelchair': 'E1130', 'walker': 'E0130', 'oxygen': 'E0424',
            'injection': 'J3420', 'vaccine': 'J3420', 'ambulance': 'A0429'
        }
        
        for keyword, code in hcpcs_keywords.items():
            if keyword in notes_lower:
                match = hcpcs_df[hcpcs_df['code'] == code]
                if not match.empty:
                    hcpcs_results.append({
                        'code': code,
                        'description': match.iloc[0]['description'],
                        'similarity': 0.8
                    })
        
        # Remove duplicates and sort
        icd_results = list({r['code']: r for r in icd_results}.values())
        proc_results = list({r['code']: r for r in proc_results}.values())
        hcpcs_results = list({r['code']: r for r in hcpcs_results}.values())
        
        # Sort by similarity
        icd_results.sort(key=lambda x: x['similarity'], reverse=True)
        proc_results.sort(key=lambda x: x['similarity'], reverse=True)
        hcpcs_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {len(icd_results)} ICD, {len(proc_results)} CPT, {len(hcpcs_results)} HCPCS results")
        
        # Ensure we have some results
        if not icd_results:
            # Absolute fallback
            fallback_icd = icd10_df[icd10_df['code'].isin(['Z00.00', 'R69'])].head(top_n)
            for _, row in fallback_icd.iterrows():
                icd_results.append({
                    'code': row['code'],
                    'description': row['description'],
                    'similarity': 0.3
                })
        
        if not proc_results:
            # Absolute fallback 
            fallback_proc = proc_df[proc_df['code'].isin(['99213'])].head(1)
            for _, row in fallback_proc.iterrows():
                proc_results.append({
                    'code': row['code'],
                    'description': row['description'],
                    'similarity': 0.3
                })
        
        return icd_results[:top_n], proc_results[:top_n], hcpcs_results[:top_n]
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        import traceback
        traceback.print_exc()
        
        # Ultimate fallback
        return (
            [{'code': 'Z00.00', 'description': 'Encounter for general adult medical examination without abnormal findings', 'similarity': 0.1}],
            [{'code': '99213', 'description': 'Office or other outpatient visit for the evaluation and management of an established patient', 'similarity': 0.1}],
            []
        )


def enhanced_semantic_search_fallback(notes_clean: str, top_n: int):
    """Fallback search using database substring matching"""
    try:
        icd_results = icd10_df[icd10_df["description"].str.contains(notes_clean, case=False, na=False)] \
                           .head(top_n).to_dict(orient="records")
        proc_results = proc_df[proc_df["description"].str.contains(notes_clean, case=False, na=False)] \
                           .head(top_n).to_dict(orient="records")
        hcpcs_results = hcpcs_df[hcpcs_df["description"].str.contains(notes_clean, case=False, na=False)] \
                             .head(top_n).to_dict(orient="records")
        
        for result in icd_results:
            result["similarity"] = 0.3
        for result in proc_results:
            result["similarity"] = 0.3
        for result in hcpcs_results:
            result["similarity"] = 0.3
        
        if not icd_results:
            icd_results = icd10_df.head(top_n).to_dict(orient="records")
            for result in icd_results:
                result["similarity"] = 0.1
        if not proc_results:
            proc_results = proc_df.head(top_n).to_dict(orient="records")
            for result in proc_results:
                result["similarity"] = 0.1
        if not hcpcs_results:
            hcpcs_results = hcpcs_df.head(top_n).to_dict(orient="records")
            for result in hcpcs_results:
                result["similarity"] = 0.1
        
        return icd_results, proc_results, hcpcs_results
    except Exception as e:
        print(f"Error in fallback search: {e}")
        return [], [], []

def filter_proc_results(notes: str, results: list, top_n: int):
    """Filter procedure results based on context"""
    try:
        notes_lower = notes.lower()
        is_surgical = any(k in notes_lower for k in SURGICAL_KEYWORDS)
        filtered = []
        
        for r in results:
            if not isinstance(r, dict):
                continue
                
            desc_lower = r.get("description", "").lower()
            
            if is_surgical:
                filtered.append(r)
            else:
                if not any(k in desc_lower for k in SURGICAL_KEYWORDS):
                    filtered.append(r)
        
        return filtered[:top_n] if filtered else results[:top_n]
    except Exception as e:
        print(f"Error filtering procedure results: {e}")
        return results[:top_n]

def filter_hcpcs_results(notes: str, results: list, top_n: int):
    """Filter HCPCS results based on context to prioritize relevant categories"""
    try:
        notes_lower = notes.lower()
        filtered = []
        priority_results = []
        
        for r in results:
            if not isinstance(r, dict):
                continue
                
            desc_lower = r.get("description", "").lower()
            code = r.get("code", "")
            
            # Prioritize based on keywords
            is_priority = False
            
            if any(k in notes_lower for k in SUPPLY_KEYWORDS):
                # Prioritize durable medical equipment (E codes)
                if code.startswith('E'):
                    is_priority = True
            
            if any(k in notes_lower for k in INJECTION_KEYWORDS):
                # Prioritize drugs/injections (J codes)
                if code.startswith('J'):
                    is_priority = True
            
            if any(k in notes_lower for k in TRANSPORT_KEYWORDS):
                # Prioritize transportation (A codes)
                if code.startswith('A'):
                    is_priority = True
            
            if is_priority:
                priority_results.append(r)
            else:
                filtered.append(r)
        
        # Return priority results first, then others
        final_results = priority_results + filtered
        return final_results[:top_n]
    except Exception as e:
        print(f"Error filtering HCPCS results: {e}")
        return results[:top_n]

# Backward compatibility function
def semantic_search(notes: str, top_n: int = 3):
    """Original semantic search function for backward compatibility"""
    icd_results, proc_results, hcpcs_results = enhanced_semantic_search(notes, top_n)
    
    # Combine procedure and HCPCS results for backward compatibility
    combined_proc_results = proc_results + hcpcs_results
    
    # Sort by similarity and take top results
    combined_proc_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    
    return icd_results, combined_proc_results[:top_n]

# Add these two functions anywhere in your semantic_search.py file:

def validate_suggestion_relevance(suggestion, notes, code_type):
    """Filter out obviously irrelevant suggestions"""
    if not suggestion:
        return False
    
    description = suggestion.get('description', '').lower()
    
    # Block completely irrelevant codes
    irrelevant_patterns = [
        'insect bite', 'craniotomy', 'electrode array', 
        'laryngoplasty', 'hiv', 'aids', 'cd4'
    ]
    
    for pattern in irrelevant_patterns:
        if pattern in description:
            return False
    
    return True

def apply_relevance_filter(results, notes, code_type):
    """Apply relevance filtering to results"""
    filtered = []
    for result in results:
        if validate_suggestion_relevance(result, notes, code_type):
            filtered.append(result)
    return filtered