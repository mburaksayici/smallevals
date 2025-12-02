"""Integration tests for pgvector using testcontainers with both SQLAlchemy and psycopg2."""

import pytest
import time
import psycopg2
from testcontainers.core.container import DockerContainer
import numpy as np
from sqlalchemy import create_engine, text

from smallevals import SmallEvalsVDBConnection, evaluate_retrievals

# PostgreSQL configuration
PG_USER = "postgres_user"
PG_PASSWORD = "postgres_password"
PG_DB = "testdb"
PG_HOST = "localhost"
PG_PORT = 5432


@pytest.fixture(scope="module")
def pgvector_container():
    """
    Starts a PostgreSQL container with pgvector extension using DockerContainer, 
    waits for it to be ready, and yields the container object.
    """
    
    # 1. Use the generic DockerContainer class
    container = DockerContainer("pgvector/pgvector:pg16")
    
    # 2. Expose the standard PostgreSQL port (5432)
    container.with_exposed_ports(5432)
    
    # 3. Set PostgreSQL configuration environment variables
    container.with_env("POSTGRES_USER", PG_USER)
    container.with_env("POSTGRES_PASSWORD", PG_PASSWORD)
    container.with_env("POSTGRES_DB", PG_DB)
    
    print("\n🚀 Starting PostgreSQL with pgvector container (DockerContainer implementation)...")
    container.start()
    
    # 4. Get connection details from the running container
    host_ip = container.get_container_host_ip()
    host_port = container.get_exposed_port(5432)
    
    # Wait for PostgreSQL to be ready via psycopg2 connection check
    max_retries = 30
    print(f"   Waiting for DB at {host_ip}:{host_port}...")
    for i in range(max_retries):
        try:
            # Try to connect to verify it's ready
            conn = psycopg2.connect(
                host=host_ip,
                port=host_port,
                database=PG_DB,
                user=PG_USER,
                password=PG_PASSWORD
            )
            # Connection successful, immediately close and break
            conn.close()
            break
        except Exception as e:
            if i == max_retries - 1:
                # If you get here, raise the error after max retries
                raise RuntimeError(f"PostgreSQL never became ready. Last error: {e}")
            time.sleep(1)  # Wait 1 second before retrying
    
    print("✅ PostgreSQL with pgvector running and ready")
    
    # Yield the container object
    yield container
    
    print("\n🛑 Stopping PostgreSQL...")
    container.stop()


@pytest.fixture(params=["sqlalchemy", "psycopg2"])
def pgvector_db(request, pgvector_container, embedding_model, qa_embeddings_parquet):
    """Create pgvector connection populated with test data.
    
    Parametrized to test both SQLAlchemy and psycopg2 connections.
    """
    connection_type = request.param
    
    # Get connection details from container
    host = pgvector_container.get_container_host_ip()
    port = pgvector_container.get_exposed_port(5432)
    
    # Build connection string
    connection_string = f"postgresql://{PG_USER}:{PG_PASSWORD}@{host}:{port}/{PG_DB}"
    
    # Create connection based on parameter
    if connection_type == "sqlalchemy":
        # Create SQLAlchemy engine
        engine = create_engine(connection_string)
        connection = engine
        print(f"\n📦 Testing with SQLAlchemy Engine")
    else:  # psycopg2
        # Create psycopg2 connection
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        )
        print(f"\n📦 Testing with psycopg2 connection")
    
    # Get embedding dimension from model
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    
    # Collection/table name
    collection_name = "test_vectors"
    
    # Setup database schema using raw SQL
    if connection_type == "sqlalchemy":
        with engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Drop table if exists
            conn.execute(text(f"DROP TABLE IF EXISTS {collection_name};"))
            
            # Create table
            conn.execute(text(f"""
                CREATE TABLE {collection_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({embedding_dim}),
                    metadata JSONB,
                    text TEXT,
                    start_index INTEGER,
                    end_index INTEGER,
                    token_count INTEGER
                );
            """))
            conn.commit()
    else:  # psycopg2
        cursor = connection.cursor()
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        connection.commit()
        
        # Drop table if exists
        cursor.execute(f"DROP TABLE IF EXISTS {collection_name};")
        connection.commit()
        
        # Create table
        cursor.execute(f"""
            CREATE TABLE {collection_name} (
                id TEXT PRIMARY KEY,
                embedding vector({embedding_dim}),
                metadata JSONB,
                text TEXT,
                start_index INTEGER,
                end_index INTEGER,
                token_count INTEGER
            );
        """)
        connection.commit()
        cursor.close()
    
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
    
    # Insert data using raw SQL
    if connection_type == "sqlalchemy":
        with engine.connect() as conn:
            insert_sql = text(f"""
                INSERT INTO {collection_name} (
                    id,
                    embedding,
                    metadata,
                    text,
                    start_index,
                    end_index,
                    token_count
                )
                VALUES (
                    :id,
                    CAST(:embedding AS vector),
                    :metadata,
                    :text,
                    :start_index,
                    :end_index,
                    :token_count
                )
            """)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # pgvector expects '[1,2,3]' format
                embedding_str = "[" + ",".join(map(str, embedding.tolist())) + "]"

                conn.execute(
                    insert_sql,
                    {
                        "id": str(i),
                        "embedding": embedding_str,
                        "metadata": "{}",
                        "text": chunk,
                        "start_index": 0,
                        "end_index": len(chunk),
                        "token_count": len(chunk.split()),
                    }
                )

            conn.commit()
    else:  # psycopg2
        cursor = connection.cursor()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedding_str = "[" + ",".join(map(str, embedding.tolist())) + "]"
            cursor.execute(
                f"""
                INSERT INTO {collection_name} (id, embedding, metadata, text, start_index, end_index, token_count)
                VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                """,
                (str(i), embedding_str, "{}", chunk, 0, len(chunk), len(chunk.split()))
            )
        connection.commit()
        cursor.close()
    
    print(f"Populated pgvector with {len(chunks)} chunks using {connection_type}")
    
    # Yield connection details for SmallEvalsVDBConnection
    yield {
        "connection": connection,
        "collection_name": collection_name,
        "connection_type": connection_type,
        "embedding_dim": embedding_dim
    }
    
    # Cleanup
    if connection_type == "sqlalchemy":
        engine.dispose()
    else:
        connection.close()


def test_pgvector_connection_setup(pgvector_db, embedding_model):
    """Test pgvector connection setup with both SQLAlchemy and psycopg2."""
    connection = pgvector_db["connection"]
    collection_name = pgvector_db["collection_name"]
    connection_type = pgvector_db["connection_type"]
    
    print(f"Testing connection setup with {connection_type}")
    
    # Create SmallEvalsVDBConnection wrapper
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=connection,
        collection=collection_name,
        embedding=embedding_model
    )
    
    assert smallevals_vdb is not None
    assert smallevals_vdb.collection_name == collection_name


def test_pgvector_query_via_wrapper(pgvector_db, embedding_model):
    """Test querying pgvector through SmallEvalsVDBConnection wrapper."""
    connection = pgvector_db["connection"]
    collection_name = pgvector_db["collection_name"]
    connection_type = pgvector_db["connection_type"]
    
    print(f"Testing query with {connection_type}")
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=connection,
        collection=collection_name,
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


def test_evaluate_retrievals_basic(pgvector_db, embedding_model):
    """Test evaluate_retrievals function with basic parameters."""
    connection = pgvector_db["connection"]
    collection_name = pgvector_db["collection_name"]
    connection_type = pgvector_db["connection_type"]
    
    print(f"Testing evaluate_retrievals with {connection_type}")
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=connection,
        collection=collection_name,
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


def test_evaluate_retrievals_with_custom_params(pgvector_db, embedding_model):
    """Test evaluate_retrievals with custom parameters."""
    connection = pgvector_db["connection"]
    collection_name = pgvector_db["collection_name"]
    connection_type = pgvector_db["connection_type"]
    
    print(f"Testing evaluate_retrievals with custom params using {connection_type}")
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=connection,
        collection=collection_name,
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


def test_pgvector_direct_search(pgvector_db, embedding_model):
    """Test direct search on pgvector connection (not through wrapper)."""
    from smallevals.vdb_integrations.pgvector_con import PgvectorConnection
    
    connection = pgvector_db["connection"]
    collection_name = pgvector_db["collection_name"]
    connection_type = pgvector_db["connection_type"]
    embedding_dim = pgvector_db["embedding_dim"]
    
    print(f"Testing direct search with {connection_type}")
    
    # Create direct PgvectorConnection
    pgvector_conn = PgvectorConnection(
        client=connection,
        collection_name=collection_name,
        embedding_model=embedding_model,
        vector_dimensions=embedding_dim
    )
    
    # Test search with query string
    results = pgvector_conn.search(query="legal framework", limit=3)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 3
    assert all("id" in r for r in results)
    assert all("text" in r for r in results)
