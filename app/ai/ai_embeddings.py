# app/ai/ai_embeddings.py

import os
import math
import numpy as np
import faiss
from llama_cpp import Llama
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from .utils import load_table
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
load_dotenv()

# --- Environment & DB ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH")
SEARCH_MODE = os.getenv("SEARCH_MODE")

# Load LLaMA for embeddings
llm = Llama(model_path=MODEL_PATH, embedding=True, n_threads=4)

ICD_FAISS_PATH = "icd10_faiss.index"
PROC_FAISS_PATH = "proc_faiss.index"

# --- Get embedding dimension dynamically ---
def get_embedding_dim():
    """Get embedding dimension by embedding a dummy text."""
    test_emb = llm.embed(["test"])
    return len(test_emb[0])

EMBEDDING_DIM = get_embedding_dim()

# --- Embedding function ---


def embed_text(texts, batch_size=64):
    """
    Generate embeddings locally using a Hugging Face model.
    Returns np.array of shape (len(texts), embedding_dim)
    """
    clean_texts = [str(t).strip() for t in texts if str(t).strip()]
    if not clean_texts:
        return np.zeros((0, 384), dtype=np.float32)  # this model outputs 384-dim vectors
    
    return model.encode(clean_texts, batch_size=batch_size, convert_to_numpy=True)


# --- Precompute embeddings & build FAISS ---
def precompute_embeddings(batch_size=500):
    """
    Precompute missing embeddings and build FAISS indices for ICD-10 and Procedural Codes.
    """
    tables = [("icd10_codes", ICD_FAISS_PATH), ("procedural_codes", PROC_FAISS_PATH)]

    for table_name, faiss_path in tables:
        df = load_table(table_name)
        missing = df[df["embedding"].isnull()]

        # Skip if embeddings and FAISS index already exist
        if missing.empty and os.path.exists(faiss_path):
            print(f"Embeddings and FAISS index already exist for {table_name}. Skipping.")
            continue

        # Compute missing embeddings
        if not missing.empty:
            print(f"Computing embeddings for {len(missing)} rows in {table_name}...")
            num_batches = math.ceil(len(missing) / batch_size)
            for i in range(num_batches):
                batch = missing.iloc[i * batch_size : (i + 1) * batch_size]
                descriptions = batch["description"].tolist()
                batch_embeddings = embed_text(descriptions, batch_size=32)

                # Update DB
                for j, row in enumerate(batch.itertuples()):
                    if j < len(batch_embeddings):
                        emb = batch_embeddings[j].tolist()
                        with engine.begin() as conn:
                            conn.execute(
                                text(f"UPDATE {table_name} SET embedding = :emb WHERE code = :code"),
                                {"emb": emb, "code": row.code}
                            )
                print(f"Batch {i + 1}/{num_batches} stored for {table_name}.")

        # Build FAISS index
        df = load_table(table_name)
        embeddings_list = [np.array(e, dtype=np.float32) for e in df["embedding"] if e is not None]

        if embeddings_list:
            embeddings = np.stack(embeddings_list)
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, faiss_path)
    
            print(f"FAISS index saved for {table_name} at {faiss_path}.")
        else:
            print(f"No embeddings found in {table_name}. FAISS index not created.")

# --- Initialize on startup if AI mode ---
if SEARCH_MODE == "ai":
    precompute_embeddings()
