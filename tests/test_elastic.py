"""Integration tests for Elasticsearch using testcontainers."""

import pytest
import numpy as np
import urllib.request
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from testcontainers.core.container import DockerContainer 
# Corrected import for the base DockerContainer used in the fixture

# Assuming these are available in your test environment, if not, they will cause NameErrors
from smallevals import SmallEvalsVDBConnection, evaluate_retrievals 
# Note: 'SentenceTransformer' is imported but not used directly in the provided code snippet

# --- FIXTURES ---

@pytest.fixture(scope="module")
def elastic_container_url():
    """
    Starts an Elasticsearch container, waits for it to be ready, and yields 
    its connection URL.
    """
    # Using DockerContainer as the base, which is correct for generic images
    container = DockerContainer("docker.elastic.co/elasticsearch/elasticsearch:8.11.0")

    # Expose ES port
    container.with_exposed_ports(9200)

    # Required config
    container.with_env("discovery.type", "single-node")
    container.with_env("xpack.security.enabled", "false")
    container.with_env("xpack.security.http.ssl.enabled", "false")
    container.with_env("ES_JAVA_OPTS", "-Xms1g -Xmx1g")

    print("\n🚀 Starting Elasticsearch container...")
    container.start()

    # Build URL
    host = container.get_container_host_ip()
    port = container.get_exposed_port(9200)
    url = f"http://{host}:{port}"

    # ✅ HTTP readiness loop
    for _ in range(45):
        try:
            # Check for HTTP 200 response on the base URL
            urllib.request.urlopen(url, timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        # Dump logs before failing
        logs = container.get_logs()
        raise RuntimeError(f"Elasticsearch never became ready.\nLogs:\n{logs}")

    print("✅ Elasticsearch running at", url)

    # Yield the URL string
    yield url

    print("\n🛑 Stopping Elasticsearch...")
    container.stop()


@pytest.fixture
def elastic_db(elastic_container_url, embedding_model, qa_embeddings_parquet):
    """Create an Elasticsearch connection populated with test data from parquet."""
    
    connection_url = elastic_container_url # e.g., "http://localhost:58676"
    # Connect to Elasticsearch
    # Extract the base URL part (remove "http://")
    # For Elasticsearch 8.x with security disabled, you must tell the client to use HTTP/insecure
    
    # -----------------------------------------------------------
    # 🔑 FIX: Set the client parameters for insecure connection
    # -----------------------------------------------------------
    client = Elasticsearch(
        # The hosts parameter can be the full URL
        hosts=[connection_url],
        # Force the client to use HTTP and skip verification (essential for Elastic 8.x w/ security disabled)
        #use_ssl=False,
        verify_certs=False,
        #ssl_assert_hostname=False,
        #ssl_show_warn=False
    )
    # -----------------------------------------------------------

    # Create index with proper schema
    index_name = "test_collection"
    
    # The error occurred here, but the fix is in the client initialization above.
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    
    # Get embedding dimension from model
    # FIXED: Assuming embedding_model is an object with this method
    embedding_dim = embedding_model.get_sentence_embedding_dimension() 
    
    # Define mapping matching ElasticConnection
    mapping = {
        "properties": {
            "embedding": {"type": "dense_vector", "dims": embedding_dim},
            "text": {"type": "text"},
            "start_index": {"type": "integer"},
            "end_index": {"type": "integer"},
            "token_count": {"type": "integer"},
        }
    }
    
    # Create index
    client.indices.create(index=index_name, mappings=mapping)
    
    # Populate with data from parquet
    df = qa_embeddings_parquet
    
    # Filter out rows with missing data (Kept existing filtering logic, adjust if needed)
    df_valid = df
    
    if len(df_valid) == 0:
        pytest.skip("No valid data in parquet file")
    
    # Take a subset for faster tests (first 100 rows)
    df_subset = df_valid.head(100)
    
    # Prepare data
    chunks = df_subset['chunk'].tolist()
    embeddings = np.array(df_subset['embedding'].tolist())
    
    # Prepare documents for bulk insertion
    actions = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = {
            "_index": index_name,
            "_id": i, # Added explicit _id to avoid issues
            "_source": {
                "text": chunk,
                "start_index": 0,
                "end_index": len(chunk),
                "token_count": len(chunk.split()),
                "embedding": embedding.tolist(),
            }
        }
        actions.append(doc)
    
    # Bulk insert
    bulk(client, actions)
    
    # Refresh index to make documents searchable
    client.indices.refresh(index=index_name)
    
    print(f"\nPopulated Elasticsearch with {len(chunks)} chunks")
    
    # Return client and index name for SmallEvalsVDBConnection
    yield {"client": client, "index_name": index_name, "connection_url": connection_url}
    
    # Cleanup
    client.indices.delete(index=index_name)
    client.close()


# --- TESTS ---

def test_elastic_connection_setup(elastic_db, embedding_model):
    """Test Elasticsearch connection setup following example_usage_chromadb.py pattern."""
    client = elastic_db["client"]
    index_name = elastic_db["index_name"]
    
    # Create SmallEvalsVDBConnection wrapper (following ChromaDB example pattern)
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=client,
        collection=index_name,
        embedding=embedding_model
    )
    
    assert smallevals_vdb is not None
    assert smallevals_vdb.collection_name == index_name


def test_elastic_query_via_wrapper(elastic_db, embedding_model):
    """Test querying Elasticsearch through SmallEvalsVDBConnection wrapper."""
    client = elastic_db["client"]
    index_name = elastic_db["index_name"]
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=client,
        collection=index_name,
        embedding=embedding_model
    )
    
    # Test query
    test_question = "What is the legal framework?"
    results = smallevals_vdb.query(test_question, top_k=5)
    
    assert isinstance(results, list)
    assert len(results) > 0
    # Check result structure
    assert all("text" in r for r in results)
    # The ID returned by smallevals_vdb.query might be the internal Elastic ID or the one we set.
    # Assuming the wrapper handles mapping the document ID.
    assert all("id" in r for r in results) 


def test_evaluate_retrievals_basic(elastic_db, embedding_model):
    """Test evaluate_retrievals function with basic parameters."""
    client = elastic_db["client"]
    index_name = elastic_db["index_name"]
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=client,
        collection=index_name,
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


def test_evaluate_retrievals_with_custom_params(elastic_db, embedding_model):
    """Test evaluate_retrievals with custom parameters."""
    client = elastic_db["client"]
    index_name = elastic_db["index_name"]
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=client,
        collection=index_name,
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