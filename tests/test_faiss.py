"""Integration tests for FAISS."""

import pytest
import numpy as np

from smallevals import SmallEvalsVDBConnection, evaluate_retrievals
from smallevals.vdb_integrations.faiss_con import FaissConnection
from sentence_transformers import SentenceTransformer

N_CHUNKS = 2


@pytest.fixture
def faiss_db(embedding_model, qa_embeddings_parquet):
    """Create a FAISS index populated via native FAISS (no custom add())"""

    import faiss
    import numpy as np

    # --- 1. Embedding dimension ---
    embedding_dim = embedding_model.get_sentence_embedding_dimension()

    # --- 2. Create FAISS index directly ---
    index = faiss.IndexFlatL2(embedding_dim)

    # --- 3. Load dataset ---
    df = qa_embeddings_parquet
    if len(df) == 0:
        pytest.skip("Empty parquet")

    df_subset = df.head(100)

    chunks = df_subset["chunk"].tolist()
    embeddings = np.vstack(df_subset["embedding"].values).astype("float32")

    # --- 4. Add directly into FAISS ---
    index.add(embeddings)

    # --- 5. Build metadata store ---
    metadatas = [
        {
            "id": str(i),
            "text": text,
            "start_index": 0,
            "end_index": len(text),
            "token_count": len(text.split()),
        }
        for i, text in enumerate(chunks)
    ]

    # --- 6. Create FaissConnection wrapper and inject index ---
    faiss_conn = FaissConnection(
        embedding_model=embedding_model,
        dimension=embedding_dim,
        index_type="Flat",
        metric="L2"
    )

    # Inject native index + metadata
    faiss_conn.index = index
    faiss_conn._chunk_data = metadatas   # SmallEvals expects this

    print(f"✅ FAISS populated with {index.ntotal} vectors")

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
    results = smallevals_vdb.search(test_question, top_k=5)
    
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
        n_chunks=N_CHUNKS,
        device=None,
        results_folder=None
    )
    
    assert result is not None
    assert "results_path" in result


def test_faiss_direct_search(faiss_db):
    """Test direct search on FAISS connection."""
    # Test with a query string
    results = faiss_db.search(query="What is the legal framework?", top_k=5)
    
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
    results = faiss_db.search(embedding=query_embedding, top_k=5)
    
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
    results = faiss_conn.search(query="test query", top_k=5)
    assert results == []
    
    # Sample on empty index should return empty list
    chunks = faiss_conn.sample_chunks(num_chunks=10)
    assert chunks == []
