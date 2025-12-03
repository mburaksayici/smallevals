"""Script to populate vector databases with case texts and create Q/A dataset.

This script:
1. Loads first 1000 case_text values from legal_text_classification.csv
2. Creates embeddings using e5-small
3. Generates Q/A pairs for chunks
4. Creates parquet file with question-answer-chunk-embeddings
5. Populates local vector databases (Chroma, Milvus, Qdrant, Weaviate, etc.)
"""
import sys
from pathlib import Path

# --- Add the project root to the path ---
# 1. Get the path of the current file (populate_dbs.py)
current_dir = Path(__file__).resolve() 
# 2. Get the parent directory (tests/)
parent_dir = current_dir.parent
# 3. Get the project root (the directory containing tests/ and smallevals/)
project_root = parent_dir.parent 

# Add the project root to sys.path
sys.path.insert(0, str(project_root))
# ---------------------------------------

import zipfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from smallevals.generation.qa_generator import QAGenerator
from smallevals.vdb_integrations.chroma_con import ChromaConnection
from smallevals.vdb_integrations.milvus_con import MilvusConnection
from smallevals.vdb_integrations.qdrant_con import QdrantConnection
from smallevals.vdb_integrations.weaviate_con import WeaviateConnection
# Paths
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
CSV_DATA_PATH = ASSETS_DIR / "legal_text_classification.csv"
TEST_VDBS_DIR = ASSETS_DIR / "test_vdbs"
PARQUET_PATH = ASSETS_DIR / "qa_embeddings.parquet"
CSV_PATH = ASSETS_DIR / "qa_embeddings.csv"

# Configuration
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"  # e5-small
NUM_QA_PAIRS = 0  # Generate Q/A for first 100 chunks only
BATCH_SIZE = 32  # Batch size for embeddings
QA_BATCH_SIZE = 8  # Batch size for QA generation


def load_case_texts() -> List[str]:
    """Load first 1000 case_text values from legal_text_classification.csv."""
    print(f"Reading case texts from {CSV_DATA_PATH}...")
    
    if not CSV_DATA_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_DATA_PATH}")
    
    # Read CSV and extract first 1000 case_text values
    df = pd.read_csv(CSV_DATA_PATH)
    
    if 'case_text' not in df.columns:
        raise ValueError(f"Column 'case_text' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Get first 1000 case_text values, filter out any NaN values
    case_texts = df['case_text'].head(300).dropna().tolist()
    
    # Filter out empty strings
    case_texts = [text.strip() for text in case_texts if text and text.strip()]
    
    print(f"Loaded {len(case_texts)} case texts")
    return case_texts


def create_embeddings(paragraphs: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Create embeddings for all paragraphs using e5-small."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Creating embeddings for {len(paragraphs)} paragraphs...")
    embeddings = model.encode(
        paragraphs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings


def generate_qa_pairs(paragraphs: List[str], num_pairs: int = NUM_QA_PAIRS) -> List[Dict[str, Any]]:
    """Generate Q/A pairs for first num_pairs chunks."""
    print(f"Generating Q/A pairs for first {num_pairs} chunks...")
    
    qa_generator = QAGenerator(device=None, batch_size=QA_BATCH_SIZE)
    
    # Prepare chunks for first num_pairs
    chunks_to_process = paragraphs[:num_pairs]
    chunks_dict = [{"text": chunk} for chunk in chunks_to_process]
    
    # Generate Q/A pairs with progress indication
    print("Processing Q/A generation (this may take a while)...")
    print(f"Processing {len(chunks_to_process)} chunks in batches of {QA_BATCH_SIZE}...")
    
    # Initialize with None to maintain index alignment
    all_qa_pairs = [None] * len(chunks_to_process)
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(chunks_to_process), QA_BATCH_SIZE), desc="Generating Q/A pairs"):
        batch_chunks = chunks_dict[i:i+QA_BATCH_SIZE]
        batch_qa = qa_generator.generate_from_chunks(batch_chunks, max_retries=1)
        
        # Match batch_qa results with their corresponding chunk indices
        # This ensures alignment even if some generations fail
        for j, qa_result in enumerate(batch_qa):
            chunk_idx = i + j
            if chunk_idx < len(all_qa_pairs) and qa_result is not None:
                all_qa_pairs[chunk_idx] = qa_result
    
    # Count successful generations
    successful_count = sum(1 for qa in all_qa_pairs if qa is not None)
    print(f"Generated {successful_count}/{len(chunks_to_process)} Q/A pairs")
    
    # Pad with None for remaining chunks (beyond num_pairs)
    all_qa_pairs = all_qa_pairs + [None] * (len(paragraphs) - len(all_qa_pairs))
    
    return all_qa_pairs


def create_dataframe(
    paragraphs: List[str],
    embeddings: np.ndarray,
    qa_pairs: List[Optional[Dict[str, Any]]]
) -> pd.DataFrame:
    """Create DataFrame with question-answer-chunk-embeddings."""
    print("Creating DataFrame...")
    
    data = []
    for i, (paragraph, embedding, qa_pair) in enumerate(zip(paragraphs, embeddings, qa_pairs)):
        row = {
            "chunk_id": f"chunk_{i}",
            "chunk": paragraph,
            "embedding": embedding.tolist(),  # Convert numpy array to list for parquet
        }
        
        if qa_pair is not None:
            row["question"] = qa_pair.get("question", "")
            row["answer"] = qa_pair.get("answer", "")
        else:
            row["question"] = None
            row["answer"] = None
        
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"Created DataFrame with {len(df)} rows and columns: {df.columns.tolist()}")
    return df


def save_parquet(df: pd.DataFrame, path: Path = PARQUET_PATH):
    """Save DataFrame to parquet file."""
    print(f"Saving DataFrame to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine='pyarrow')
    print(f"✅ Saved parquet file: {path}")


def save_csv(df: pd.DataFrame, path: Path = CSV_PATH):
    """Save DataFrame to CSV file (without embedding column)."""
    print(f"Saving DataFrame to CSV: {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create copy without embedding column
    df_csv = df.copy()
    if 'embedding' in df_csv.columns:
        df_csv = df_csv.drop(columns=['embedding'])
    
    # Reorder columns: chunk_id, chunk, question, answer
    column_order = ['chunk_id', 'chunk', 'question', 'answer']
    # Add any other columns that exist
    for col in df_csv.columns:
        if col not in column_order:
            column_order.append(col)
    
    # Reorder to match desired order
    df_csv = df_csv[[col for col in column_order if col in df_csv.columns]]
    
    df_csv.to_csv(path, index=False)
    print(f"✅ Saved CSV file: {path}")
    print(f"   Columns: {', '.join(df_csv.columns.tolist())}")
    print(f"   Rows: {len(df_csv)}")


def setup_vdb_directories():
    """Create directory structure for vector databases."""
    print("Setting up vector database directories...")
    
    vdb_types = ["chroma", "milvus", "qdrant", "weaviate"]
    
    for vdb_type in vdb_types:
        vdb_dir = TEST_VDBS_DIR / vdb_type
        vdb_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created directory: {vdb_dir}")


def populate_chroma(
    paragraphs: List[str],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer
):
    """Populate ChromaDB with all chunks."""
   
    
    print("\n" + "="*60)
    print("Populating ChromaDB...")
    print("="*60)
    
    chroma_path = TEST_VDBS_DIR / "chroma"
    
    try:
        # Initialize ChromaDB connection
        chroma_conn = ChromaConnection(
            path=str(chroma_path),
            collection_name="paragraphs",
            embedding_model=embedding_model
        )
        
        # Prepare chunks for insertion
        print(f"Adding {len(paragraphs)} chunks to ChromaDB...")
        
        # Add chunks in batches
        batch_size = 1000
        for i in tqdm(range(0, len(paragraphs), batch_size), desc="Adding chunks"):
            batch_paragraphs = paragraphs[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_ids = [f"chunk_{j}" for j in range(i, min(i+batch_size, len(paragraphs)))]
            batch_metadatas = [{"chunk_id": f"chunk_{j}"} for j in range(i, min(i+batch_size, len(paragraphs)))]
            
            # Use collection.add directly
            chroma_conn.collection.add(
                ids=batch_ids,
                documents=batch_paragraphs,
                embeddings=batch_embeddings.tolist(),
                metadatas=batch_metadatas
            )
        
        print(f"✅ ChromaDB populated: {chroma_path}")
    except Exception as e:
        print(f"❌ Failed to populate ChromaDB: {e}")
        import traceback
        traceback.print_exc()




def main():
    """Main function to run the entire pipeline."""
    print("="*60)
    print("Database Population Script")
    print("="*60)
    
    # Step 1: Load case texts from CSV
    paragraphs = load_case_texts()
    print(len(paragraphs))
    # Step 2: Create embeddings
    embeddings = create_embeddings(paragraphs)
    
    # Step 3: Generate Q/A pairs
    # qa_pairs = generate_qa_pairs(paragraphs)
    qa_pairs = [None] * len(embeddings)
    # Step 4: Create DataFrame
    df = create_dataframe(paragraphs, embeddings, qa_pairs=qa_pairs)
    
    # Step 5: Save parquet file
    save_parquet(df)
    
    # Step 6: Save CSV file (without embeddings)
    save_csv(df)
    
    # Step 7: Setup VDB directories
    setup_vdb_directories()
    
    # Step 8: Load embedding model for VDBs
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Step 9: Populate vector databases
    populate_chroma(paragraphs, embeddings, embedding_model)
    
    print("\n" + "="*60)
    print("✅ Database population complete!")
    print("="*60)
    print(f"Parquet file: {PARQUET_PATH}")
    print(f"Vector databases: {TEST_VDBS_DIR}")


if __name__ == "__main__":
    main()

