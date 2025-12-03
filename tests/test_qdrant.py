"""Integration tests for Qdrant using testcontainers."""

import pytest
import numpy as np
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from smallevals import SmallEvalsVDBConnection, evaluate_retrievals
from sentence_transformers import SentenceTransformer
N_CHUNKS = 2

@pytest.fixture(scope="module")
def qdrant_container():
    container = DockerContainer("qdrant/qdrant:latest")

    # Required ports
    container.with_exposed_ports(6333, 6334)

    print("🚀 Starting Qdrant container…")
    container.start()

    # Wait for Qdrant to be ready
    wait_for_logs(container, "Qdrant gRPC listening", timeout=60)

    print("✅ Qdrant started")

    yield container

    print("🛑 Stopping Qdrant…")
    container.stop()


@pytest.fixture
def qdrant_db(qdrant_container, embedding_model, qa_embeddings_parquet):
    """Create a Qdrant connection populated with test data from parquet."""
    # Get connection details from container
    host = qdrant_container.get_container_host_ip()
    port = int(qdrant_container.get_exposed_port(6333))
    
    # Connect to Qdrant
    client = QdrantClient(host=host, port=port)
    
    # Create collection with proper schema
    collection_name = "TestCollection"
    
    # Delete collection if it exists (for clean test state)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection doesn't exist, which is fine
    
    # Get embedding dimension from model
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE
        )
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
    
    # Prepare points for insertion
    points = [
        PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload={
                "text": chunks[idx],
                "start_index": 0,
                "end_index": len(chunks[idx]),
                "token_count": len(chunks[idx].split()),
            }
        )
        for idx in range(len(chunks))
    ]
    
    # Insert data
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Populated Qdrant with {len(chunks)} chunks")
    
    # Return connection details for SmallEvalsVDBConnection
    yield {"client": client, "collection_name": collection_name, "host": host, "port": port}
    
    # Cleanup
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def test_qdrant_connection_setup(qdrant_db, embedding_model):
    """Test Qdrant connection setup following example_usage_chromadb.py pattern."""
    collection_name = qdrant_db["collection_name"]
    client = qdrant_db["client"]
    
    # Create SmallEvalsVDBConnection wrapper (following ChromaDB example pattern)
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=client,
        collection=collection_name,
        embedding=embedding_model
    )
    
    assert smallevals_vdb is not None
    assert smallevals_vdb.collection_name == collection_name


def test_qdrant_query_via_wrapper(qdrant_db, embedding_model):
    """Test querying Qdrant through SmallEvalsVDBConnection wrapper."""
    collection_name = qdrant_db["collection_name"]
    client = qdrant_db["client"]
    
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


def test_evaluate_retrievals_basic(qdrant_db, embedding_model):
    """Test evaluate_retrievals function with basic parameters."""
    collection_name = qdrant_db["collection_name"]
    client = qdrant_db["client"]
    
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


def test_evaluate_retrievals_with_custom_params(qdrant_db, embedding_model):
    """Test evaluate_retrievals with custom parameters."""
    collection_name = qdrant_db["collection_name"]
    client = qdrant_db["client"]
    
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

