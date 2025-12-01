"""Integration tests for ChromaDB."""

import pytest
from smallevals.eval.metrics import calculate_retrieval_metrics_full


def test_chromadb_query(chroma_db, sample_qa_pairs):
    """Test querying ChromaDB."""
    # Get first QA pair
    qa_pair = sample_qa_pairs[0]
    question = qa_pair["question"]
    
    # Query ChromaDB
    results = chroma_db.query(question, top_k=5)
    
    assert len(results) > 0
    assert all("text" in r for r in results)
    assert all("id" in r or "metadata" in r for r in results)


def test_chromadb_evaluation(chroma_db, sample_qa_pairs):
    """Test evaluating ChromaDB retrieval."""
    # Use first 10 QA pairs for faster test
    test_qa_pairs = sample_qa_pairs[:10]
    
    # Query vector DB for each question
    retrieval_results = []
    for qa_pair in test_qa_pairs:
        question = qa_pair.get("question", "")
        if not question:
            retrieval_results.append([])
            continue
        retrieved = chroma_db.query(question, top_k=5)
        retrieval_results.append(retrieved)
    
    # Calculate metrics
    metrics_result = calculate_retrieval_metrics_full(
        test_qa_pairs, retrieval_results, top_k=5
    )
    metrics = metrics_result["aggregated"]
    
    assert "precision@5" in metrics
    assert "recall@5" in metrics
    assert "mrr" in metrics
    assert "hit_rate@5" in metrics
    assert "per_sample" in metrics_result
    
    # Check that metrics are valid (between 0 and 1)
    assert 0.0 <= metrics["precision@5"] <= 1.0
    assert 0.0 <= metrics["recall@5"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert 0.0 <= metrics["hit_rate@5"] <= 1.0
    
    # Check per_sample structure
    assert len(metrics_result["per_sample"]) == len(test_qa_pairs)


def test_chromadb_metrics_with_chunk_ids(chroma_db, sample_qa_pairs):
    """Test that metrics use chunk ID matching."""
    # Get a QA pair
    qa_pair = sample_qa_pairs[0]
    question = qa_pair["question"]
    chunk_id = qa_pair["chunk_id"]
    
    # Query ChromaDB
    retrieved = chroma_db.query(question, top_k=5)
    
    # Check that retrieved chunks have IDs
    assert all("id" in r or ("metadata" in r and "chunk_id" in r.get("metadata", {})) for r in retrieved)
    
    # Calculate metrics
    retrieval_results = [retrieved]
    qa_dataset = [qa_pair]
    
    metrics_result = calculate_retrieval_metrics_full(
        qa_dataset, retrieval_results, top_k=5
    )
    
    # Check that metrics were calculated
    assert "aggregated" in metrics_result
    assert len(metrics_result["per_sample"]) == 1


def test_chromadb_sample_chunks(chroma_db):
    """Test sampling chunks from ChromaDB."""
    sampled = chroma_db.sample_chunks(10)
    
    assert len(sampled) <= 10
    assert all("text" in chunk for chunk in sampled)
    assert all("id" in chunk or "metadata" in chunk for chunk in sampled)


def test_chromadb_with_smallevals_api(chroma_db, embedding_model, test_data_dir):
    """Test ChromaDB using the README API pattern with SmallEvalsVDBConnection."""
    import chromadb
    from smallevals import SmallEvalsVDBConnection, evaluate_retrievals
    
    # Use tests/assets/test_vdbs/chroma as the persistent client folder (as per README pattern)
    chroma_path = test_data_dir / "test_vdbs" / "chroma"
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Create SmallEvalsVDBConnection wrapper as shown in README
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=chroma_client,
        collection=chroma_db.collection_name,
        embedding=embedding_model
    )
    
    # Test that the wrapper works
    assert smallevals_vdb is not None
    assert smallevals_vdb.collection_name == chroma_db.collection_name
    
    # Test query through wrapper
    test_question = "test query"
    results = smallevals_vdb.query(test_question, top_k=5)
    assert isinstance(results, list)
    
    # Note: evaluate_retrievals generates QA pairs, so we skip full evaluation
    # in unit tests to avoid requiring the QA generation model
    # This test just verifies the API connection works

