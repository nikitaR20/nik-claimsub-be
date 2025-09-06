# app/ai/semantic_search.py
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
# Load environment variables from .env file (only in local dev)
load_dotenv()

# Get the database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
# Handle Heroku's postgres URLs (optional: convert if needed)
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# --- Connect to PostgreSQL ---
engine = create_engine(DATABASE_URL)

# --- Load ICD-10 and Procedure Codes ---
def load_table(table_name):
    query = text(f"SELECT code, description FROM {table_name}")
    df = pd.read_sql(query, engine)

    # Clean descriptions
    df["description"] = df["description"].astype(str).fillna("").str.strip()
    df = df[df["description"] != ""]  # remove empty rows

    if df.empty:
        raise ValueError(f"No valid descriptions found in table: {table_name}")

    return df

icd10_df = load_table("icd10_codes")
proc_df = load_table("procedural_codes")

# --- Build Vectorizer and Matrix ---
def create_vectorizer_and_matrix(df):
    descriptions = df["description"].astype(str).fillna("").str.strip()
    descriptions = descriptions[descriptions != ""]

    if descriptions.empty:
        raise ValueError("No valid descriptions to process in TF-IDF.")

    vectorizer = TfidfVectorizer(stop_words=None)  # don't remove stop words for medical text
    matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, matrix

icd10_vectorizer, icd10_matrix = create_vectorizer_and_matrix(icd10_df)
proc_vectorizer, proc_matrix = create_vectorizer_and_matrix(proc_df)

# --- Semantic Search Function ---
def semantic_search(notes: str, top_n: int = 3):
    """
    Returns top N most similar ICD-10 and procedure codes for the given coverage notes.
    """
    notes = notes.strip().lower()
    if not notes:
        return [], []

    # ICD-10 similarity
    note_vec_icd = icd10_vectorizer.transform([notes])
    icd10_scores = cosine_similarity(note_vec_icd, icd10_matrix).flatten()
    top_icd10_idx = icd10_scores.argsort()[-top_n:][::-1]
    icd10_results = icd10_df.iloc[top_icd10_idx][["code", "description"]].to_dict(orient="records")

    # Procedure codes similarity
    note_vec_proc = proc_vectorizer.transform([notes])
    proc_scores = cosine_similarity(note_vec_proc, proc_matrix).flatten()
    top_proc_idx = proc_scores.argsort()[-top_n:][::-1]
    proc_results = proc_df.iloc[top_proc_idx][["code", "description"]].to_dict(orient="records")

    return icd10_results, proc_results
