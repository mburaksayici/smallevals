"""Pytest configuration and fixtures for EvalVD tests."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evalvd.vdb_integrations.chroma_con import ChromaConnection
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test assets directory."""
    return Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def qa_embeddings_parquet(test_data_dir):
    """Load QA embeddings parquet file."""
    parquet_path = test_data_dir / "qa_embeddings.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Test data not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


@pytest.fixture(scope="session")
def embedding_model():
    """Load embedding model for tests."""
    return SentenceTransformer('all-MiniLM-L6-v2')


@pytest.fixture
def chroma_db(test_data_dir, embedding_model, qa_embeddings_parquet):
    """Create or use existing ChromaDB instance populated with test data."""
    # Check if test_vdbs/chroma exists
    test_vdbs_dir = test_data_dir / "test_vdbs"
    chroma_path = test_vdbs_dir / "chroma"
    
    # Use existing database if it exists, otherwise use tmp_path
    use_existing = chroma_path.exists() and chroma_path.is_dir()
    
    if use_existing:
        # Use existing ChromaDB - get_or_create_collection will use existing collection
        db_path = str(chroma_path)
        print(f"Using existing ChromaDB at {db_path}")
    else:
        # Create temporary database for this test run
        import tempfile
        import atexit
        temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
        db_path = temp_dir
        print(f"Creating temporary ChromaDB at {db_path}")
        
        def cleanup():
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        atexit.register(cleanup)
    
    # Create ChromaDB connection (get_or_create_collection will use existing if available)
    chroma_conn = ChromaConnection(
        path=db_path,
        collection_name="test_collection",
        embedding_model=embedding_model
    )
    
    # Check if collection already has data
    collection_count = chroma_conn.collection.count()
    
    if collection_count == 0:
        # Populate with data from parquet
        print(f"Populating ChromaDB with test data...")
        df = qa_embeddings_parquet
        
        # Filter out rows with missing questions
        df_valid = df[df['question'].notna() & df['chunk'].notna()].copy()
        
        if len(df_valid) == 0:
            pytest.skip("No valid data in parquet file")
        
        # Take a subset for faster tests (first 100 rows)
        df_subset = df_valid.head(100)
        
        # Prepare data
        chunks = df_subset['chunk'].tolist()
        chunk_ids = df_subset['chunk_id'].tolist()
        embeddings = np.array(df_subset['embedding'].tolist())
        
        # Add to ChromaDB
        chroma_conn.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[{"chunk_id": cid} for cid in chunk_ids]
        )
        print(f"Populated ChromaDB with {len(chunk_ids)} chunks")
    else:
        print(f"Using existing ChromaDB collection with {collection_count} chunks")
    
    yield chroma_conn
    
    # Only cleanup if we created a temporary database
    if not use_existing:
        import shutil
        if Path(db_path).exists():
            shutil.rmtree(db_path)


@pytest.fixture
def sample_qa_pairs(qa_embeddings_parquet):
    """Create sample QA pairs from parquet data."""
    df = qa_embeddings_parquet
    
    # Filter valid rows
    df_valid = df[df['question'].notna() & df['answer'].notna() & df['chunk_id'].notna()].copy()
    
    if len(df_valid) == 0:
        pytest.skip("No valid QA pairs in parquet file")
    
    # Convert to list of dicts
    qa_pairs = []
    for _, row in df_valid.head(50).iterrows():  # Use first 50 for tests
        qa_pairs.append({
            "question": row['question'],
            "answer": row['answer'],
            "chunk_id": row['chunk_id'],
            "passage": row['chunk'],  # For backward compatibility
        })
    
    return qa_pairs

