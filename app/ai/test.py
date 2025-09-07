import os
from app.ai.ai_embeddings import embed_text, ICD_FAISS_PATH, PROC_FAISS_PATH, precompute_embeddings
from app.ai.semantic_search import semantic_search, SEARCH_MODE
from app.ai.utils import load_table

# --- Step 1: Load tables and verify ---
icd10_df = load_table("icd10_codes")
proc_df = load_table("procedural_codes")

print(f"ICD-10 codes rows: {len(icd10_df)}")
print(f"Procedural codes rows: {len(proc_df)}")

# --- Step 2: Test DB search ---
SEARCH_MODE = "db"
query = "diabetes"
icd_db_results, proc_db_results = semantic_search(query, top_n=3)
print(f"\n--- DB Search Results for '{query}' ---")
print("ICD Results:", icd_db_results)
print("Procedure Results:", proc_db_results)
print("ICD embeddings exist:", not icd10_df["embedding"].isnull().all())
print("Procedure embeddings exist:", not proc_df["embedding"].isnull().all())
print("ICD FAISS file exists:", os.path.exists(ICD_FAISS_PATH))
print("Procedure FAISS file exists:", os.path.exists(PROC_FAISS_PATH))

# --- Step 3: Test FAISS search ---
SEARCH_MODE = "ai"

# Precompute embeddings if missing
for df, path in [(icd10_df, ICD_FAISS_PATH), (proc_df, PROC_FAISS_PATH)]:
    if df["embedding"].isnull().any() or not os.path.exists(path):
        print(f"Missing embeddings or index file for {path}. Precomputing...")
        precompute_embeddings()

query = "heart attack"
icd_ai_results, proc_ai_results = semantic_search(query, top_n=3)
print(f"\n--- FAISS Search Results for '{query}' ---")
print("ICD Results:", icd_ai_results)
print("Procedure Results:", proc_ai_results)
