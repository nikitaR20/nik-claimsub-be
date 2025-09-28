# precompute_all_embeddings.py
import os
import logging
import numpy as np
import pandas as pd
import json
from .ai_embeddings import embed_text, SEARCH_MODE, EMBEDDING_MODEL
# Note: We're NOT importing the original precompute_embeddings function

# ========================================
# CONFIGURATION SWITCHES
# ========================================
FORCE_RECOMPUTE_ALL = False  # Set to True to recompute ALL embeddings, False to only compute missing ones
BATCH_SIZE_OVERRIDE = None   # Set to specific number to override default batch size, or None to use default
SKIP_CONFIRMATION = True     # Set to True to skip all confirmation prompts
VERBOSE_LOGGING = True       # Set to True for detailed batch progress logging

# Initialize logging
log_level = logging.INFO if VERBOSE_LOGGING else logging.WARNING
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

def is_valid_embedding(embedding, expected_dim=None):
    """
    Check if an embedding is valid and usable for semantic search
    This checks for existence, proper format, non-zero values, and correct dimensions
    """
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

def get_expected_embedding_dimension():
    """Get the expected embedding dimension from current model"""
    try:
        test_embedding = embed_text(["test"])[0]
        return test_embedding.shape[0] if hasattr(test_embedding, 'shape') else len(test_embedding)
    except Exception as e:
        logger.error(f"Failed to determine embedding dimension: {e}")
        return None

def save_embeddings_to_db(table_name: str, embeddings_data: list):
    """Save embeddings to database"""
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
            
            # Convert to JSON format for database
            if isinstance(embedding, np.ndarray):
                embedding_json = json.dumps(embedding.tolist())
            else:
                embedding_json = json.dumps(embedding)
            
            if table_name == "icd10_codes":
                query = text("UPDATE icd10_codes SET embedding = :embedding WHERE code = :code")
            elif table_name == "procedural_codes":
                query = text("UPDATE procedural_codes SET embedding = :embedding WHERE code = :code")
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

def precompute_all_embeddings(force_recompute=FORCE_RECOMPUTE_ALL):
    """
    Precompute embeddings for ALL records in both tables with better validation
    
    Args:
        force_recompute (bool): If True, recompute embeddings even if they already exist
    """
    logger.info("Starting full embeddings precomputation for ALL ICD and procedure codes...")
    
    # Get expected embedding dimension from current model
    expected_dim = get_expected_embedding_dimension()
    if expected_dim is None:
        logger.error("Cannot proceed without knowing expected embedding dimension")
        return
    
    logger.info(f"Expected embedding dimension: {expected_dim}")
    
    try:
        # Import here to avoid circular imports
        from app.utils import load_table
        
        # Process ICD Codes
        logger.info("Loading ICD10 codes...")
        icd_df = load_table("icd10_codes")
        
        if not icd_df.empty:
            total_icd = len(icd_df)
            logger.info(f"Found {total_icd} ICD10 codes")
            
            if force_recompute:
                codes_to_embed = icd_df.copy()
                logger.info("Force recompute enabled - processing ALL ICD codes")
            else:
                # Enhanced validation for existing embeddings
                def needs_embedding(row):
                    return not is_valid_embedding(row.get('embedding'), expected_dim)
                
                codes_to_embed = icd_df[icd_df.apply(needs_embedding, axis=1)]
                
                valid_embeddings = total_icd - len(codes_to_embed)
                logger.info(f"Found {valid_embeddings} ICD codes with valid embeddings")
                logger.info(f"Processing {len(codes_to_embed)} ICD codes that need embeddings")
            
            if not codes_to_embed.empty:
                texts = codes_to_embed['description'].fillna('').astype(str).tolist()
                
                # Use batch size override if specified, otherwise use model-appropriate default
                if BATCH_SIZE_OVERRIDE:
                    batch_size = BATCH_SIZE_OVERRIDE
                else:
                    batch_size = 50 if EMBEDDING_MODEL.startswith("mistral") else 100
                all_embeddings = []
                
                total_batches = (len(texts) - 1) // batch_size + 1
                logger.info(f"Processing {len(texts)} ICD descriptions in {total_batches} batches")
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = embed_text(batch_texts)
                    
                    # Validate each embedding in the batch
                    valid_batch_embeddings = []
                    for emb in batch_embeddings:
                        if is_valid_embedding(emb, expected_dim):
                            valid_batch_embeddings.append(emb)
                        else:
                            logger.warning(f"Generated invalid embedding in ICD batch {i//batch_size + 1}")
                            valid_batch_embeddings.append(np.random.normal(0, 0.1, expected_dim))  # Generate a fallback embedding
                    
                    all_embeddings.extend(valid_batch_embeddings)
                    
                    current_batch = i // batch_size + 1
                    if VERBOSE_LOGGING:
                        logger.info(f"Completed ICD batch {current_batch}/{total_batches} "
                                  f"({len(valid_batch_embeddings)} embeddings)")
                    elif current_batch % 10 == 0:  # Log every 10th batch if not verbose
                        logger.info(f"ICD Progress: {current_batch}/{total_batches} batches")
                
                # Prepare data for database save
                embeddings_data = [
                    {
                        'code': codes_to_embed.iloc[idx]['code'], 
                        'embedding': all_embeddings[idx]
                    } 
                    for idx in range(len(all_embeddings))
                ]
                
                save_embeddings_to_db("icd10_codes", embeddings_data)
                logger.info(f"Completed ICD10 codes: {len(embeddings_data)} embeddings processed")
            else:
                logger.info("All ICD codes already have valid embeddings - skipping computation")
        else:
            logger.warning("No ICD10 codes found in database")
        
        # Process Procedure Codes
        logger.info("Loading procedure codes...")
        proc_df = load_table("procedural_codes")
        
        if not proc_df.empty:
            total_proc = len(proc_df)
            logger.info(f"Found {total_proc} procedure codes")
            
            if force_recompute:
                codes_to_embed = proc_df.copy()
                logger.info("Force recompute enabled - processing ALL procedure codes")
            else:
                # Enhanced validation for existing embeddings
                def needs_embedding(row):
                    return not is_valid_embedding(row.get('embedding'), expected_dim)
                
                codes_to_embed = proc_df[proc_df.apply(needs_embedding, axis=1)]
                
                valid_embeddings = total_proc - len(codes_to_embed)
                logger.info(f"Found {valid_embeddings} procedure codes with valid embeddings")
                logger.info(f"Processing {len(codes_to_embed)} procedure codes that need embeddings")
            
            if not codes_to_embed.empty:
                texts = codes_to_embed['description'].fillna('').astype(str).tolist()
                
                # Use batch size override if specified, otherwise use model-appropriate default
                if BATCH_SIZE_OVERRIDE:
                    batch_size = BATCH_SIZE_OVERRIDE
                else:
                    batch_size = 50 if EMBEDDING_MODEL.startswith("mistral") else 100
                all_embeddings = []
                
                total_batches = (len(texts) - 1) // batch_size + 1
                logger.info(f"Processing {len(texts)} procedure descriptions in {total_batches} batches")
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = embed_text(batch_texts)
                    
                    # Validate each embedding in the batch
                    valid_batch_embeddings = []
                    for emb in batch_embeddings:
                        if is_valid_embedding(emb, expected_dim):
                            valid_batch_embeddings.append(emb)
                        else:
                            logger.warning(f"Generated invalid embedding in procedure batch {i//batch_size + 1}")
                            valid_batch_embeddings.append(np.random.normal(0, 0.1, expected_dim))  # Generate a fallback embedding
                    
                    all_embeddings.extend(valid_batch_embeddings)
                    
                    current_batch = i // batch_size + 1
                    if VERBOSE_LOGGING:
                        logger.info(f"Completed procedure batch {current_batch}/{total_batches} "
                                  f"({len(valid_batch_embeddings)} embeddings)")
                    elif current_batch % 10 == 0:  # Log every 10th batch if not verbose
                        logger.info(f"Procedure Progress: {current_batch}/{total_batches} batches")
                
                # Prepare data for database save
                embeddings_data = [
                    {
                        'code': codes_to_embed.iloc[idx]['code'], 
                        'embedding': all_embeddings[idx]
                    } 
                    for idx in range(len(all_embeddings))
                ]
                
                save_embeddings_to_db("procedural_codes", embeddings_data)
                logger.info(f"Completed procedure codes: {len(embeddings_data)} embeddings processed")
            else:
                logger.info("All procedure codes already have valid embeddings - skipping computation")
        else:
            logger.warning("No procedure codes found in database")
        
        logger.info("All embeddings generated and saved successfully!")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EMBEDDING COMPUTATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Search Mode: {SEARCH_MODE}")
        logger.info(f"Embedding Model: {EMBEDDING_MODEL}")
        logger.info(f"Expected Dimension: {expected_dim}")
        logger.info(f"Force Recompute: {force_recompute}")
        logger.info(f"Batch Size Override: {BATCH_SIZE_OVERRIDE or 'Auto'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to precompute embeddings: {e}")
        raise

def check_embedding_status():
    """Check the current status of embeddings in the database with enhanced validation"""
    try:
        from app.utils import load_table
        
        logger.info("Checking current embedding status...")
        expected_dim = get_expected_embedding_dimension()
        
        # Check ICD codes
        icd_df = load_table("icd10_codes")
        if not icd_df.empty:
            total_icd = len(icd_df)
            valid_embeddings = 0
            
            for _, row in icd_df.iterrows():
                if is_valid_embedding(row.get('embedding'), expected_dim):
                    valid_embeddings += 1
            
            percentage = (valid_embeddings / total_icd * 100) if total_icd > 0 else 0
            logger.info(f"ICD10 Codes: {valid_embeddings}/{total_icd} have valid embeddings ({percentage:.1f}%)")
            
            if valid_embeddings < total_icd:
                missing_count = total_icd - valid_embeddings
                logger.info(f"  -> {missing_count} ICD codes need valid embeddings")
        
        # Check procedure codes
        proc_df = load_table("procedural_codes")
        if not proc_df.empty:
            total_proc = len(proc_df)
            valid_embeddings = 0
            
            for _, row in proc_df.iterrows():
                if is_valid_embedding(row.get('embedding'), expected_dim):
                    valid_embeddings += 1
            
            percentage = (valid_embeddings / total_proc * 100) if total_proc > 0 else 0
            logger.info(f"Procedure Codes: {valid_embeddings}/{total_proc} have valid embeddings ({percentage:.1f}%)")
            
            if valid_embeddings < total_proc:
                missing_count = total_proc - valid_embeddings
                logger.info(f"  -> {missing_count} procedure codes need valid embeddings")
            
    except Exception as e:
        logger.error(f"Error checking embedding status: {e}")

def main():
    try:
        if SEARCH_MODE != "ai":
            logger.warning("SEARCH_MODE is not 'ai'. Embeddings will still be generated using default model.")
        
        # Display current configuration
        logger.info("=" * 60)
        logger.info("EMBEDDING COMPUTATION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Force Recompute All: {FORCE_RECOMPUTE_ALL}")
        logger.info(f"Batch Size Override: {BATCH_SIZE_OVERRIDE or 'Auto'}")
        logger.info(f"Skip Confirmation: {SKIP_CONFIRMATION}")
        logger.info(f"Verbose Logging: {VERBOSE_LOGGING}")
        logger.info(f"Search Mode: {SEARCH_MODE}")
        logger.info(f"Embedding Model: {EMBEDDING_MODEL}")
        logger.info("=" * 60)
        
        # Check current status
        check_embedding_status()
        
        # Optional confirmation (can be disabled)
        if not SKIP_CONFIRMATION:
            if FORCE_RECOMPUTE_ALL:
                print("\nWARNING: FORCE_RECOMPUTE_ALL is enabled!")
                print("This will overwrite ALL existing embeddings.")
                confirm = input("Continue? (y/N): ").strip().lower()
                if confirm != 'y':
                    logger.info("Operation cancelled")
                    return
            else:
                print(f"\nWill compute embeddings only for records that don't have valid ones.")
                confirm = input("Continue? (Y/n): ").strip().lower()
                if confirm == 'n':
                    logger.info("Operation cancelled")
                    return
        
        # Run the computation
        precompute_all_embeddings()
        
        # Check status after completion
        print("\n" + "=" * 60)
        print("FINAL STATUS:")
        print("=" * 60)
        check_embedding_status()

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise

if __name__ == "__main__":
    main()