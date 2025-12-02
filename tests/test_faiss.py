"""Integration tests for FAISS."""

import pytest
import numpy as np

from smallevals import SmallEvalsVDBConnection, evaluate_retrievals
from smallevals.vdb_integrations.faiss_con import FaissConnection
from sentence_transformers import SentenceTransformer


@pytest.fixture
def faiss_db(embedding_model, qa_embeddings_parquet):
    """Create a FAISS index populated with test data from parquet."""
    # Get embedding dimension from model
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    
    # Create FAISS connection
    faiss_conn = FaissConnection(
        embedding_model=embedding_model,
        dimension=embedding_dim,
        index_type="Flat",
        metric="L2"
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
    
    # Prepare metadata
    metadatas = [
        {
            "start_index": 0,
            "end_index": len(c),
            "token_count": len(c.split()),
        }
        for c in chunks
    ]
    
    # Add data to FAISS
    faiss_conn.add(texts=chunks, embeddings=embeddings, metadatas=metadatas)
    
    print(f"Populated FAISS with {len(chunks)} chunks")
    
    yield faiss_conn


def test_faiss_connection_setup(faiss_db, embedding_model):
    """Test FAISS connection setup following example_usage_chromadb.py pattern."""
    # Create SmallEvalsVDBConnection wrapper (following ChromaDB example pattern)
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=faiss_db,
        collection=None,  # FAISS doesn't use collections
        embedding=embedding_model
    )
    
    assert smallevals_vdb is not None
    assert smallevals_vdb.connection == faiss_db


def test_faiss_query_via_wrapper(faiss_db, embedding_model):
    """Test querying FAISS through SmallEvalsVDBConnection wrapper."""
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=faiss_db,
        collection=None,
        embedding=embedding_model
    )
    
    # Test query
    test_question = "What is the legal framework?"
    results = smallevals_vdb.query(test_question, top_k=5)
    
    assert isinstance(results, list)
    assert len(results) > 0
    # Check result structure
    assert all("text" in r for r in results)
    assert all("id" in r for r in results)


def test_evaluate_retrievals_basic(faiss_db, embedding_model):
    """Test evaluate_retrievals function with basic parameters."""
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=faiss_db,
        collection=None,
        embedding=embedding_model
    )
    
    # Run evaluation with small number of chunks for faster tests
    result = evaluate_retrievals(
        connection=smallevals_vdb,
        top_k=10,
        n_chunks=20,  # Small number for faster tests
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


def test_evaluate_retrievals_with_custom_params(faiss_db, embedding_model):
    """Test evaluate_retrievals with custom parameters."""
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=faiss_db,
        collection=None,
        embedding=embedding_model
    )
    
    # Test with different top_k
    result = evaluate_retrievals(
        connection=smallevals_vdb,
        top_k=5,
        n_chunks=10,
        device=None,
        results_folder=None
    )
    
    assert result is not None
    assert "results_path" in result


def test_faiss_direct_search(faiss_db):
    """Test direct search on FAISS connection."""
    # Test with a query string
    results = faiss_db.search(query="What is the legal framework?", limit=5)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)
    assert all("id" in r for r in results)


def test_faiss_search_with_embedding(faiss_db, embedding_model):
    """Test FAISS search with pre-computed embedding."""
    # Generate an embedding
    query_text = "What is the legal framework?"
    query_embedding = embedding_model.encode(query_text)
    
    # Search with embedding
    results = faiss_db.search(embedding=query_embedding, limit=5)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert all("text" in r for r in results)


def test_faiss_sample_chunks(faiss_db):
    """Test sampling random chunks from FAISS index."""
    # Sample 10 chunks
    chunks = faiss_db.sample_chunks(num_chunks=10)
    
    assert isinstance(chunks, list)
    assert len(chunks) == 10
    assert all("text" in c for c in chunks)
    assert all("id" in c for c in chunks)


def test_faiss_empty_index():
    """Test FAISS with an empty index."""
    from sentence_transformers import SentenceTransformer
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    
    faiss_conn = FaissConnection(
        embedding_model=embedding_model,
        dimension=embedding_dim,
    )
    
    # Search on empty index should return empty list
    results = faiss_conn.search(query="test query", limit=5)
    assert results == []
    
    # Sample on empty index should return empty list
    chunks = faiss_conn.sample_chunks(num_chunks=10)
    assert chunks == []


def test_faiss_add_vectors(embedding_model):
    """Test adding vectors to FAISS index."""
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    
    faiss_conn = FaissConnection(
        embedding_model=embedding_model,
        dimension=embedding_dim,
    )
    
    # Add some test data
    texts = ["This is a test", "Another test", "Third test"]
    embeddings = np.random.rand(3, embedding_dim).astype(np.float32)
    metadatas = [
        {"start_index": 0, "end_index": 14, "token_count": 4},
        {"start_index": 0, "end_index": 12, "token_count": 2},
        {"start_index": 0, "end_index": 10, "token_count": 2},
    ]
    
    faiss_conn.add(texts=texts, embeddings=embeddings, metadatas=metadatas)
    
    # Check that vectors were added
    assert faiss_conn.index.ntotal == 3
    assert len(faiss_conn._chunk_data) == 3
    
    # Check that we can search
    results = faiss_conn.search(embedding=embeddings[0], limit=1)
    assert len(results) == 1
    assert results[0]["text"] == "This is a test"

