#!/usr/bin/env python3
import sys
import os

# FIX: Disable problematic debugger modules
try:
    import pdb
    pdb.set_trace = lambda: None
except:
    pass

# Prevent bytecode generation that can cause issues
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import os
import numpy as np
import pandas as pd
from typing import List, Optional
from dotenv import load_dotenv
import logging
from sentence_transformers import SentenceTransformer
import json
import requests
from llama_cpp import Llama

# Load environment variables
load_dotenv()

# Configuration
SEARCH_MODE = os.getenv("SEARCH_MODE", "db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers")
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# Mistral Local Configuration
MISTRAL_MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH", "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
MISTRAL_API_URL = os.getenv("MISTRAL_API_URL", "http://localhost:8080")
MISTRAL_N_CTX = int(os.getenv("MISTRAL_N_CTX", "4096"))
MISTRAL_N_THREADS = int(os.getenv("MISTRAL_N_THREADS", "4"))

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
ICD_FAISS_PATH = os.path.join(EMBEDDINGS_DIR, "icd_faiss.index")
PROC_FAISS_PATH = os.path.join(EMBEDDINGS_DIR, "proc_faiss.index")
HCPCS_FAISS_PATH = os.path.join(EMBEDDINGS_DIR, "hcpcs_faiss.index")

# Create embeddings directory if it doesn't exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
_sentence_transformer_model = None
_mistral_model = None

def get_sentence_transformer_model():
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            logger.info(f"Loading SentenceTransformer model: {SENTENCE_TRANSFORMER_MODEL}")
            _sentence_transformer_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {e}")
            # Fallback to a basic model
            try:
                _sentence_transformer_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                logger.info("Loaded fallback model: paraphrase-MiniLM-L3-v2")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {e2}")
                raise Exception("Could not load any SentenceTransformer model")
    return _sentence_transformer_model

def get_mistral_model():
    global _mistral_model
    if _mistral_model is None:
        try:
            if not os.path.exists(MISTRAL_MODEL_PATH):
                raise FileNotFoundError(f"Mistral model file not found: {MISTRAL_MODEL_PATH}")
            logger.info(f"Loading Mistral model: {MISTRAL_MODEL_PATH}")
            _mistral_model = Llama(
                model_path=MISTRAL_MODEL_PATH,
                n_ctx=MISTRAL_N_CTX,
                n_threads=MISTRAL_N_THREADS,
                verbose=False,
                embedding=True
            )
            logger.info("Mistral model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Mistral model: {e}")
            raise
    return _mistral_model

def is_valid_embedding(embedding, expected_dim=None):
    """Check if an embedding is valid and usable for semantic search"""
    if embedding is None:
        return False
    
    # Handle string format (JSON)
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
    
    # Handle array format
    if isinstance(embedding, (list, np.ndarray)):
        if len(embedding) == 0:
            return False
        
        # Convert to numpy array for easier checking
        if isinstance(embedding, list):
            try:
                embedding = np.array(embedding, dtype=np.float32)
            except (ValueError, TypeError):
                return False
        
        # Check for all zeros or NaN values (indicates bad embedding)
        if np.allclose(embedding, 0) or np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False
        
        # Check dimension if specified
        if expected_dim is not None and len(embedding) != expected_dim:
            return False
        
        return True
    
    return False

def embed_text_mistral_local(texts: List[str]) -> List[np.ndarray]:
    try:
        model = get_mistral_model()
        embeddings = []
        for text in texts:
            cleaned_text = text.replace("\n", " ").strip()
            if not cleaned_text:
                cleaned_text = "medical diagnosis"
            
            embedding = model.embed(cleaned_text)

            # Convert to np.ndarray if needed
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)

            # If 2D (tokens x dim), do mean pooling to get single vector
            if embedding.ndim == 2:
                embedding = embedding.mean(axis=0)

            # Validate the embedding
            if not is_valid_embedding(embedding):
                logger.warning(f"Generated invalid embedding for text: {cleaned_text[:50]}...")
                # Create a fallback embedding
                embedding = np.random.normal(0, 0.1, size=embedding.shape).astype(np.float32)

            embeddings.append(embedding)

        logger.info(f"Generated {len(embeddings)} embeddings using local Mistral")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating Mistral embeddings: {e}")
        logger.info("Falling back to SentenceTransformers")
        return embed_text_sentence_transformers(texts)

def embed_text_mistral_api(texts: List[str]) -> List[np.ndarray]:
    try:
        embeddings = []
        for text in texts:
            cleaned_text = text.replace("\n", " ").strip()
            if not cleaned_text:
                cleaned_text = "medical diagnosis"
                
            response = requests.post(
                f"{MISTRAL_API_URL}/embeddings",
                json={"content": cleaned_text},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                embedding_array = np.array(embedding, dtype=np.float32)
                
                # Validate the embedding
                if not is_valid_embedding(embedding_array):
                    logger.warning(f"Generated invalid embedding via API for text: {cleaned_text[:50]}...")
                    # Create a fallback embedding
                    embedding_array = np.random.normal(0, 0.1, size=len(embedding)).astype(np.float32)
                
                embeddings.append(embedding_array)
            else:
                logger.warning(f"API call failed with status {response.status_code}")
                # Create a fallback embedding
                embeddings.append(np.random.normal(0, 0.1, size=384).astype(np.float32))
                
        logger.info(f"Generated {len(embeddings)} embeddings using Mistral API")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating Mistral API embeddings: {e}")
        logger.info("Falling back to SentenceTransformers")
        return embed_text_sentence_transformers(texts)

def embed_text_with_mistral_similarity(texts: List[str]) -> List[np.ndarray]:
    try:
        medical_concepts = [
            "pain", "surgery", "diagnosis", "treatment", "patient", "medical", "clinical",
            "disease", "symptom", "procedure", "therapy", "condition", "acute", "chronic",
            "infection", "inflammation", "injury", "disorder", "syndrome", "examination"
        ]
        embeddings = []
        for text in texts:
            embedding = []
            text_lower = text.lower() if text else "medical diagnosis"
            
            for concept in medical_concepts:
                count = text_lower.count(concept)
                score = min(count * 0.3, 1.0)
                embedding.append(score)
            
            text_hash = hash(text if text else "default") % 1000
            for i in range(20):
                embedding.append((text_hash + i) % 100 / 100.0)
            
            arr = np.array(embedding, dtype=np.float32)
            
            # Ensure the embedding is not all zeros
            if np.allclose(arr, 0):
                arr = np.random.normal(0, 0.1, size=len(embedding)).astype(np.float32)
            else:
                # Normalize
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
            
            embeddings.append(arr)
        
        logger.info(f"Generated {len(embeddings)} concept-based embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error in concept-based embedding: {e}")
        return [np.random.normal(0, 0.1, 44).astype(np.float32) for _ in texts]

def embed_text_sentence_transformers(texts: List[str]) -> List[np.ndarray]:
    try:
        model = get_sentence_transformer_model()
        
        # Clean and validate input texts
        cleaned_texts = []
        for text in texts:
            if not text or not text.strip():
                cleaned_texts.append("medical diagnosis")
            else:
                cleaned_texts.append(text.replace("\n", " ").strip())
        
        embeddings = model.encode(cleaned_texts, convert_to_tensor=False)
        embeddings_list = []
        
        for i, e in enumerate(embeddings):
            if not isinstance(e, np.ndarray):
                e = np.array(e, dtype=np.float32)
            else:
                e = e.astype(np.float32)
            
            # Validate each embedding
            if not is_valid_embedding(e):
                logger.warning(f"Generated invalid embedding for text: {cleaned_texts[i][:50]}...")
                # Create a fallback embedding
                e = np.random.normal(0, 0.1, size=e.shape).astype(np.float32)
            
            embeddings_list.append(e)
        
        logger.info(f"Generated {len(embeddings_list)} embeddings using SentenceTransformers")
        return embeddings_list
    except Exception as e:
        logger.error(f"Error generating SentenceTransformer embeddings: {e}")
        raise

def embed_text(texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    if isinstance(texts, str):
        texts = [texts]
    
    try:
        if EMBEDDING_MODEL == "mistral-local":
            return embed_text_mistral_local(texts)
        elif EMBEDDING_MODEL == "mistral-api":
            return embed_text_mistral_api(texts)
        elif EMBEDDING_MODEL == "mistral-similarity":
            return embed_text_with_mistral_similarity(texts)
        else:
            return embed_text_sentence_transformers(texts)
    except Exception as e:
        logger.error(f"Error in embed_text: {e}")
        if EMBEDDING_MODEL.startswith("mistral"):
            try:
                return embed_text_sentence_transformers(texts)
            except:
                # Last resort: generate random embeddings
                fallback_dim = 4096 if EMBEDDING_MODEL == "mistral-local" else 384
                return [np.random.normal(0, 0.1, fallback_dim).astype(np.float32) for _ in texts]
        else:
            # Last resort: generate random embeddings
            return [np.random.normal(0, 0.1, 384).astype(np.float32) for _ in texts]

def load_table_with_embeddings(table_name: str):
    try:
        from app.database import get_db
        from sqlalchemy import text
        db = next(get_db())
        
        # Updated to support HCPCS
        if table_name == "icd10_codes":
            query = "SELECT code, description, embedding FROM icd10_codes WHERE embedding IS NOT NULL"
        elif table_name == "procedural_codes":
            query = "SELECT code, description, embedding FROM procedural_codes WHERE embedding IS NOT NULL"
        elif table_name == "hcpcs_codes":  # New HCPCS support
            query = "SELECT code, description, embedding FROM hcpcs_codes WHERE embedding IS NOT NULL"
        else:
            raise ValueError(f"Unknown table name: {table_name}")
            
        result = db.execute(text(query)).fetchall()
        data = []
        for row in result:
            embedding = None
            if row[2]:
                try:
                    if isinstance(row[2], str):
                        parsed = json.loads(row[2])
                        if is_valid_embedding(parsed):
                            embedding = np.array(parsed, dtype=np.float32)
                    elif isinstance(row[2], (list, np.ndarray)):
                        if is_valid_embedding(row[2]):
                            embedding = np.array(row[2], dtype=np.float32)
                except:
                    embedding = None
            data.append({'code': row[0], 'description': row[1], 'embedding': embedding})
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records from {table_name}")
        return df
    except Exception as e:
        logger.error(f"Error loading table {table_name}: {e}")
        return pd.DataFrame(columns=['code', 'description', 'embedding'])

def save_embeddings_to_db(table_name: str, embeddings_data: List[dict]):
    try:
        from app.database import get_db
        from sqlalchemy import text
        db = next(get_db())
        
        saved_count = 0
        for item in embeddings_data:
            code = item['code']
            embedding = item['embedding']
            
            # Validate embedding before saving
            if not is_valid_embedding(embedding):
                logger.warning(f"Skipping invalid embedding for code {code}")
                continue
            
            embedding_json = json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
            
            # Updated to support HCPCS
            if table_name == "icd10_codes":
                query = text("UPDATE icd10_codes SET embedding = :embedding WHERE code = :code")
            elif table_name == "procedural_codes":
                query = text("UPDATE procedural_codes SET embedding = :embedding WHERE code = :code")
            elif table_name == "hcpcs_codes":  # New HCPCS support
                query = text("UPDATE hcpcs_codes SET embedding = :embedding WHERE code = :code")
            else:
                continue
                
            result = db.execute(query, {"embedding": embedding_json, "code": code})
            if result.rowcount > 0:
                saved_count += 1
        
        db.commit()
        logger.info(f"Saved {saved_count} valid embeddings for {table_name}")
    except Exception as e:
        logger.error(f"Error saving embeddings to {table_name}: {e}")
        raise

def precompute_embeddings():
    logger.info("Starting precomputation of embeddings...")
    try:
        from app.utils import load_table
        
        # Get expected dimension for validation
        expected_dim = None
        try:
            test_embedding = embed_text(["test"])[0]
            expected_dim = test_embedding.shape[0] if hasattr(test_embedding, 'shape') else len(test_embedding)
        except Exception as e:
            logger.error(f"Failed to determine expected embedding dimension: {e}")
        
        # Define tables to process - now includes HCPCS
        tables_to_process = [
            ("icd10_codes", "ICD"),
            ("procedural_codes", "Procedure"),
            ("hcpcs_codes", "HCPCS")  # New HCPCS table
        ]
        
        for table_name, display_name in tables_to_process:
            logger.info(f"Processing {display_name} codes...")
            df = load_table(table_name)
            
            if df.empty:
                logger.warning(f"No data found in {table_name} - skipping")
                continue
            
            # Use the enhanced validation
            def needs_embedding(row):
                return not is_valid_embedding(row.get('embedding'), expected_dim)
            
            codes_to_embed = df[df.apply(needs_embedding, axis=1)]
            
            if not codes_to_embed.empty:
                texts = codes_to_embed['description'].fillna('').astype(str).tolist()
                batch_size = 50 if EMBEDDING_MODEL.startswith("mistral") else 100
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = embed_text(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"Processed {display_name} batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                embeddings_data = [
                    {'code': codes_to_embed.iloc[idx]['code'], 'embedding': all_embeddings[idx]} 
                    for idx in range(len(all_embeddings))
                ]
                save_embeddings_to_db(table_name, embeddings_data)
            else:
                logger.info(f"All {display_name} codes already have valid embeddings")
                
        logger.info("Embeddings precomputation completed successfully!")
    except Exception as e:
        logger.error(f"Error in precompute_embeddings: {e}")
        raise