"""Integration tests for ChromaDB using disk-based storage."""

import pytest
import numpy as np
import chromadb
from pathlib import Path
import tempfile
import shutil

from smallevals import SmallEvalsVDBConnection, evaluate_retrievals
from smallevals.vdb_integrations.chroma_con import ChromaConnection
from sentence_transformers import SentenceTransformer

N_CHUNKS = 2
@pytest.fixture
def chromadb_db(embedding_model, qa_embeddings_parquet):
    """Create a ChromaDB connection populated with test data from parquet."""
    # Create temporary directory for ChromaDB
    temp_dir = tempfile.mkdtemp()
    chroma_path = Path(temp_dir)
    
    try:
        collection_name = "TestCollection"
        
        # Create ChromaDB connection using ChromaConnection (which sets up embedding function properly)
        chroma_conn = ChromaConnection(
            path=str(chroma_path),
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        
        # Populate with data from parquet
        df = qa_embeddings_parquet
        
        # Filter out rows with missing data
        df_valid = df
        
        if len(df_valid) == 0:
            pytest.skip("No valid data in parquet file")
        
        # Take a subset for faster tests (first 100 rows)
        df_subset = df_valid.head(100)
        
        # Prepare data
        chunks = df_subset['chunk'].tolist()
        embeddings = np.array(df_subset['embedding'].tolist())
        
        # Generate IDs for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Prepare metadata
        metadatas = [
            {
                "start_index": 0,
                "end_index": len(c),
                "token_count": len(c.split())
            }
            for c in chunks
        ]
        
        # Insert data
        chroma_conn.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        print(f"Populated ChromaDB with {len(chunks)} chunks")
        
        # Return client and collection name for SmallEvalsVDBConnection
        yield {"client": chroma_conn.client, "collection_name": collection_name, "path": str(chroma_path)}
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp directory: {e}")


def test_chromadb_connection_setup(chromadb_db, embedding_model):
    """Test ChromaDB connection setup following example_usage_chromadb.py pattern."""
    client = chromadb_db["client"]
    collection_name = chromadb_db["collection_name"]
    
    # Create SmallEvalsVDBConnection wrapper (following ChromaDB example pattern)
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=client,
        collection=collection_name,
        embedding=embedding_model
    )
    
    assert smallevals_vdb is not None
    assert smallevals_vdb.collection_name == collection_name


def test_chromadb_query_via_wrapper(chromadb_db, embedding_model):
    """Test querying ChromaDB through SmallEvalsVDBConnection wrapper."""
    client = chromadb_db["client"]
    collection_name = chromadb_db["collection_name"]
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=client,
        collection=collection_name,
        embedding=embedding_model
    )
    
    # Test query
    test_question = "What is the legal framework?"
    results = smallevals_vdb.search(test_question, top_k=5)
    
    assert isinstance(results, list)
    assert len(results) > 0
    # Check result structure
    assert all("text" in r for r in results)
    assert all("id" in r for r in results)


def test_evaluate_retrievals_basic(chromadb_db, embedding_model):
    """Test evaluate_retrievals function with basic parameters."""
    client = chromadb_db["client"]
    collection_name = chromadb_db["collection_name"]
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=client,
        collection=collection_name,
        embedding=embedding_model
    )
    
    # Run evaluation with small number of chunks for faster tests
    result = evaluate_retrievals(
        connection=smallevals_vdb,
        top_k=10,
        n_chunks=N_CHUNKS,  # Small number for faster tests
        device=None,
        results_folder=None
    )
    
    # Check result structure
    assert result is not None
    assert "results_path" in result
    assert isinstance(result["results_path"], (str, type(None)))
    
    # Check that evaluation completed
    if result.get("results_path"):
        from pathlib import Path
        results_path = Path(result["results_path"])
        assert results_path.exists() or results_path.parent.exists()


def test_evaluate_retrievals_with_custom_params(chromadb_db, embedding_model):
    """Test evaluate_retrievals with custom parameters."""
    client = chromadb_db["client"]
    collection_name = chromadb_db["collection_name"]
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=client,
        collection=collection_name,
        embedding=embedding_model
    )
    
    # Test with different top_k
    result = evaluate_retrievals(
        connection=smallevals_vdb,
        top_k=5,
        n_chunks=N_CHUNKS,
        device=None,
        results_folder=None
    )
    
    assert result is not None
    assert "results_path" in result
