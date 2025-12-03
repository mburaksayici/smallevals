"""Integration tests for Milvus using testcontainers."""

import pytest
import numpy as np
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from smallevals import SmallEvalsVDBConnection, evaluate_retrievals
from sentence_transformers import SentenceTransformer
N_CHUNKS = 2

@pytest.fixture(scope="module")
def milvus_container():
    container = DockerContainer("milvusdb/milvus:v2.4.0")

    # Required ports
    container.with_exposed_ports(19530, 9091, 2379, 9000)

    # Required envs
    container.with_env("ETCD_USE_EMBED", "true")
    container.with_env("COMMON_STORAGETYPE", "local")

    # Correct startup command
    container.with_command("milvus run standalone")

    print("🚀 Starting Milvus container…")
    container.start()

    # Wait long enough
    wait_for_logs(container, "Proxy successfully started", timeout=60)

    print("✅ Milvus started")

    yield container

    print("🛑 Stopping Milvus…")
    container.stop()


@pytest.fixture
def milvus_db(milvus_container, embedding_model, qa_embeddings_parquet):
    """Create a Milvus connection populated with test data from parquet."""
    # Get connection details from container
    host = milvus_container.get_container_host_ip()
    port = int(milvus_container.get_exposed_port(19530))
    
    # Connect to Milvus
    alias = "test_connection"
    connections.connect(
        alias=alias,
        host=host,
        port=port,
    )
    
    # Create collection with proper schema
    collection_name = "TestCollection"
    
    # Delete collection if it exists (for clean test state)
    if utility.has_collection(collection_name, using=alias):
        utility.drop_collection(collection_name, using=alias)
    
    # Get embedding dimension from model
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    
    # Define schema matching MilvusConnection
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="start_index", dtype=DataType.INT64),
        FieldSchema(name="end_index", dtype=DataType.INT64),
        FieldSchema(name="token_count", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    ]
    schema = CollectionSchema(fields, description="Test Collection for SmallEvals")
    
    # Create collection
    collection = Collection(name=collection_name, schema=schema, using=alias)
    
    # Create index for vector field
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
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
    
    # Prepare entities for insertion
    entities = [
        chunks,  # text field
        [0] * len(chunks),  # start_index
        [len(c) for c in chunks],  # end_index
        [len(c.split()) for c in chunks],  # token_count
        embeddings.tolist(),  # embedding
    ]
        
    # Insert data
    collection.insert(entities)
    collection.flush()

    # Create index AFTER data is present
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    # Ensure index is fully built
    collection.flush()

    # Load collection for search AFTER index exists
    collection.load()

    print(f"✅ Populated Milvus with {len(chunks)} chunks")

    # Return connection alias and collection name for SmallEvalsVDBConnection
    yield {"alias": alias, "collection_name": collection_name, "host": host, "port": port}

    # Cleanup
    collection.release()
    connections.disconnect(alias=alias)


def test_milvus_connection_setup(milvus_db, embedding_model):
    """Test Milvus connection setup following example_usage_chromadb.py pattern."""
    collection_name = milvus_db["collection_name"]
    alias = milvus_db["alias"]
    
    # Get the collection for passing to SmallEvalsVDBConnection
    collection = Collection(collection_name, using=alias)
    
    # Create SmallEvalsVDBConnection wrapper (following ChromaDB example pattern)
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=collection,
        collection=collection_name,
        embedding=embedding_model
    )
    
    assert smallevals_vdb is not None
    assert smallevals_vdb.collection_name == collection_name


def test_milvus_query_via_wrapper(milvus_db, embedding_model):
    """Test querying Milvus through SmallEvalsVDBConnection wrapper."""
    collection_name = milvus_db["collection_name"]
    alias = milvus_db["alias"]

    # Get the collection for passing to SmallEvalsVDBConnection
    collection = Collection(collection_name, using=alias)
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=collection,
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


def test_evaluate_retrievals_basic(milvus_db, embedding_model):
    """Test evaluate_retrievals function with basic parameters."""
    collection_name = milvus_db["collection_name"]
    alias = milvus_db["alias"]
    
    # Get the collection for passing to SmallEvalsVDBConnection
    collection = Collection(collection_name, using=alias)
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=collection,
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


def test_evaluate_retrievals_with_custom_params(milvus_db, embedding_model):
    """Test evaluate_retrievals with custom parameters."""
    collection_name = milvus_db["collection_name"]
    alias = milvus_db["alias"]
    
    # Get the collection for passing to SmallEvalsVDBConnection
    collection = Collection(collection_name, using=alias)
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=collection,
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

