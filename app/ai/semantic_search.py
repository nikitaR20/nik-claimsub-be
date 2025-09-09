import os
import numpy as np
import faiss
from .utils import load_table
from .ai_embeddings import embed_text, ICD_FAISS_PATH, PROC_FAISS_PATH, precompute_embeddings
from dotenv import load_dotenv
#SEARCH_MODE = "faiss"  # default to FAISS search

load_dotenv()
SEARCH_MODE = os.getenv("SEARCH_MODE")
# --- Load tables ---
icd10_df = load_table("icd10_codes")
proc_df = load_table("procedural_codes")

# Convert embeddings to numpy arrays
for df in [icd10_df, proc_df]:
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32) if x else None)

# --- Function to load or build FAISS index safely ---
def load_faiss_index(table_name, df, path):
    # Ensure missing embeddings are computed first
    if df["embedding"].isnull().any():
        print(f"Missing embeddings in {table_name}. Computing now...")
        precompute_embeddings()
        df = load_table(table_name)
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32) if x else None)

    embeddings_list = [e for e in df["embedding"] if e is not None]
    if not embeddings_list:
        raise ValueError(f"No embeddings found in {table_name} to build FAISS index!")

    embeddings = np.stack(embeddings_list)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save index for future runs
    faiss.write_index(index, path)
    print(f"FAISS index saved for {table_name} at {path}")
    return index

# --- Initialize FAISS indices ---
if os.path.exists(ICD_FAISS_PATH):
    print(f"Loading existing FAISS index: {ICD_FAISS_PATH}")
    icd_index = faiss.read_index(ICD_FAISS_PATH)
else:
    icd_index = load_faiss_index("icd10_codes", icd10_df, ICD_FAISS_PATH)

if os.path.exists(PROC_FAISS_PATH):
    print(f"Loading existing FAISS index: {PROC_FAISS_PATH}")
    proc_index = faiss.read_index(PROC_FAISS_PATH)
else:
    proc_index = load_faiss_index("procedural_codes", proc_df, PROC_FAISS_PATH)

# --- Semantic search ---
SURGICAL_KEYWORDS = ["surgery", "decompression", "arthrodesis", "fusion", "resection"]

def semantic_search(notes: str, top_n: int = 3):
    notes = notes.strip()
    if not notes:
        return [], []

    notes_clean = " ".join(notes.lower().split())  # normalize spacing & lowercase

    # --- Simple DB Substring Search (Fallback) ---
    if SEARCH_MODE == "db":
        icd_results = icd10_df[icd10_df["description"].str.contains(notes_clean, case=False, na=False)] \
                           .head(top_n).to_dict(orient="records")
        proc_results = proc_df[proc_df["description"].str.contains(notes_clean, case=False, na=False)] \
                           .head(top_n).to_dict(orient="records")
        
        # fallback: return top 1 if empty
        if not icd_results:
            icd_results = icd10_df.head(1).to_dict(orient="records")
        if not proc_results:
            proc_results = proc_df.head(1).to_dict(orient="records")
        
        return icd_results, proc_results

    # --- AI/Embedding Search using FAISS ---
    try:
        query_vector = embed_text([notes_clean])[0].astype(np.float32)
        faiss.normalize_L2(query_vector.reshape(1, -1))
    except Exception:
        # fallback: return top 1 from DB if embedding fails
        return icd10_df.head(1).to_dict(orient="records"), proc_df.head(1).to_dict(orient="records")

    def search_faiss(df, index):
        D, I = index.search(query_vector.reshape(1, -1), top_n)
        results = []
        for idx, sim in zip(I[0], D[0]):
            if idx == -1:
                continue
            row = df.iloc[idx]
            results.append({
                "code": row.code,
                "description": row.description,
                "similarity": float(sim)
            })
        # fallback if nothing found
        if not results:
            results = df.head(1).to_dict(orient="records")
        return results

    icd_results = search_faiss(icd10_df, icd_index)
    proc_results = search_faiss(proc_df, proc_index)

    # --- Filter procedure codes based on note context ---
    def filter_proc_results(notes, results):
        notes_lower = notes.lower()
        is_surgical = any(k in notes_lower for k in SURGICAL_KEYWORDS)
        filtered = []
        for r in results:
            desc_lower = r["description"].lower()
            if is_surgical:
                filtered.append(r)
            else:
                # skip surgical codes for non-surgical notes
                if not any(k in desc_lower for k in SURGICAL_KEYWORDS):
                    filtered.append(r)
        return filtered[:top_n]

    proc_results = filter_proc_results(notes, proc_results)

    # Sort by similarity descending
    proc_results = sorted(proc_results, key=lambda x: x["similarity"], reverse=True)
    icd_results = sorted(icd_results, key=lambda x: x["similarity"], reverse=True)

    return icd_results, proc_results
