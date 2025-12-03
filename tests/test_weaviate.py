"""Integration tests for Weaviate using testcontainers."""

import pytest
import numpy as np
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
import weaviate
from weaviate.classes.config import Configure, DataType, Property

from smallevals import SmallEvalsVDBConnection, evaluate_retrievals
from sentence_transformers import SentenceTransformer

N_CHUNKS = 2
@pytest.fixture(scope="module")
def weaviate_container():
    """Start Weaviate container for tests."""
    container = DockerContainer("semitechnologies/weaviate:latest")
    container.with_exposed_ports(8080, 50051)
    container.with_env("AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED", "true")
    container.with_env("PERSISTENCE_DATA_PATH", "/var/lib/weaviate")
    container.with_env("QUERY_DEFAULTS_LIMIT", "25")
    container.with_env("DEFAULT_VECTORIZER_MODULE", "none")
    container.with_env("CLUSTER_HOSTNAME", "node1")
    
    container.start()
    
    # Wait for Weaviate to be ready
    wait_for_logs(container, "Serving weaviate", timeout=60)
    
    yield container
    
    container.stop()


@pytest.fixture
def weaviate_db(weaviate_container, embedding_model, qa_embeddings_parquet):
    """Create a Weaviate client populated with test data from parquet."""
    # Get connection details from container
    http_host = weaviate_container.get_container_host_ip()
    http_port = int(weaviate_container.get_exposed_port(8080))
    grpc_port = int(weaviate_container.get_exposed_port(50051))
    
    # Create raw Weaviate client (v4 API)
    weaviate_client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=False,
        grpc_host=http_host,
        grpc_port=grpc_port,
        grpc_secure=False,
    )
    
    # Create collection with proper schema
    collection_name = "TestCollection"
    
    # Delete collection if it exists (for clean test state)
    if weaviate_client.collections.exists(collection_name):
        weaviate_client.collections.delete(collection_name)
    
    # Create collection with schema matching WeaviateConnection expectations
    weaviate_client.collections.create(
        name=collection_name,
        vector_index_config=Configure.VectorIndex.hnsw(),
        properties=[
            Property(
                name="text",
                data_type=DataType.TEXT,
                description="The text content of the chunk",
            ),
            Property(
                name="start_index",
                data_type=DataType.INT,
                description="The start index of the chunk in the original text",
            ),
            Property(
                name="end_index",
                data_type=DataType.INT,
                description="The end index of the chunk in the original text",
            ),
            Property(
                name="token_count",
                data_type=DataType.INT,
                description="The number of tokens in the chunk",
            ),
            Property(
                name="chunk_type",
                data_type=DataType.TEXT,
                description="The type of the chunk",
            ),
        ],
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
    chunk_ids = df_subset['chunk_id'].tolist()
    embeddings = np.array(df_subset['embedding'].tolist())
    
    # Add to Weaviate using batch insert (v4 API)
    collection = weaviate_client.collections.get(collection_name)
    
    with collection.batch.dynamic() as batch:
        for chunk, chunk_id, embedding in zip(chunks, chunk_ids, embeddings):
            batch.add_object(
                properties={
                    "text": chunk,
                    "start_index": 0,
                    "end_index": len(chunk),
                    "token_count": len(chunk.split()),
                    "chunk_type": "",
                },
                vector=embedding.tolist(),
            )
    
    print(f"Populated Weaviate with {len(chunks)} chunks")
    
    # Return raw client (not WeaviateConnection wrapper)
    yield weaviate_client
    
    # Cleanup
    weaviate_client.close()


def test_weaviate_connection_setup(weaviate_db, embedding_model):
    """Test Weaviate connection setup following example_usage_chromadb.py pattern."""
    COLLECTION_NAME = "TestCollection"
    
    # Create SmallEvalsVDBConnection wrapper (following ChromaDB example pattern)
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=weaviate_db,
        collection=COLLECTION_NAME,
        embedding=embedding_model
    )
    
    assert smallevals_vdb is not None
    assert smallevals_vdb.collection_name == COLLECTION_NAME


def test_weaviate_query_via_wrapper(weaviate_db, embedding_model):
    """Test querying Weaviate through SmallEvalsVDBConnection wrapper."""
    COLLECTION_NAME = "TestCollection"
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=weaviate_db,
        collection=COLLECTION_NAME,
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


def test_evaluate_retrievals_basic(weaviate_db, embedding_model):
    """Test evaluate_retrievals function with basic parameters."""
    COLLECTION_NAME = "TestCollection"

    smallevals_vdb = SmallEvalsVDBConnection(
        connection=weaviate_db,
        collection=COLLECTION_NAME,
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


def test_evaluate_retrievals_with_custom_params(weaviate_db, embedding_model):
    """Test evaluate_retrievals with custom parameters."""
    COLLECTION_NAME = "TestCollection"

    smallevals_vdb = SmallEvalsVDBConnection(
        connection=weaviate_db,
        collection=COLLECTION_NAME,
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
