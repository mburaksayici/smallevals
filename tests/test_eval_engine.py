"""Integration tests for evaluation engine."""

import pytest
import json
from pathlib import Path
import tempfile

from smallevals.eval.engine import generate_qa_from_vectordb, evaluate_vectordb, evaluate_rag
from smallevals.exceptions import ValidationError


def test_generate_qa_from_vectordb_validation():
    """Test input validation for generate_qa_from_vectordb."""
    # Mock vector DB
    class MockVDB:
        def sample_chunks(self, num_chunks):
            return [{"text": f"chunk {i}", "id": f"chunk_{i}"} for i in range(num_chunks)]
    
    mock_vdb = MockVDB()
    
    # Test invalid num_chunks
    with pytest.raises(ValidationError):
        generate_qa_from_vectordb(mock_vdb, num_chunks=0)
    
    with pytest.raises(ValidationError):
        generate_qa_from_vectordb(mock_vdb, num_chunks=-1)
    
    # Test invalid batch_size
    with pytest.raises(ValidationError):
        generate_qa_from_vectordb(mock_vdb, num_chunks=10, batch_size=0)


def test_evaluate_vectordb_validation():
    """Test input validation for evaluate_vectordb."""
    # Mock vector DB
    class MockVDB:
        def query(self, question, top_k=5):
            return [{"text": "result", "id": "chunk_1"}]
    
    mock_vdb = MockVDB()
    
    # Test invalid top_k
    qa_pairs = [{"question": "q1", "answer": "a1", "chunk_id": "chunk_1", "passage": "text1"}]
    
    with pytest.raises(ValidationError):
        evaluate_vectordb(qa_pairs, mock_vdb, top_k=0)
    
    with pytest.raises(ValidationError):
        evaluate_vectordb(qa_pairs, mock_vdb, top_k=-1)
    
    # Test invalid QA pair structure
    invalid_qa = [{"question": "q1"}]  # Missing chunk_id and passage
    
    with pytest.raises(ValidationError):
        evaluate_vectordb(invalid_qa, mock_vdb, top_k=5)
    
    # Test empty dataset
    with pytest.raises(ValidationError):
        evaluate_vectordb([], mock_vdb, top_k=5)
    
    # Test non-existent file
    with pytest.raises(ValidationError):
        evaluate_vectordb("/nonexistent/file.jsonl", mock_vdb, top_k=5)


def test_evaluate_vectordb_with_file(chroma_db, sample_qa_pairs, tmp_path):
    """Test evaluate_vectordb with JSONL file input."""
    # Create temporary JSONL file
    jsonl_file = tmp_path / "test_qa.jsonl"
    with open(jsonl_file, "w") as f:
        for qa in sample_qa_pairs[:5]:
            f.write(json.dumps(qa) + "\n")
    
    # Evaluate
    metrics = evaluate_vectordb(
        qa_dataset=str(jsonl_file),
        vectordb=chroma_db,
        top_k=5
    )
    
    assert "precision@5" in metrics
    assert "per_sample" in metrics
    assert len(metrics["per_sample"]) == 5


def test_evaluate_rag_validation():
    """Test input validation for evaluate_rag."""
    # Mock vector DB
    class MockVDB:
        def query(self, question, top_k=5):
            return [{"text": "result", "id": "chunk_1"}]
    
    mock_vdb = MockVDB()
    qa_pairs = [{"question": "q1", "answer": "a1", "chunk_id": "chunk_1", "passage": "text1"}]
    
    # Test invalid rag_pipeline (not callable)
    with pytest.raises(ValidationError):
        evaluate_rag(qa_pairs, mock_vdb, rag_pipeline="not a function", top_k=5)
    
    # Test valid rag_pipeline
    def rag_func(question):
        return "answer"
    
    result = evaluate_rag(qa_pairs, mock_vdb, rag_pipeline=rag_func, top_k=5)
    assert "retrieval" in result
    assert "generation" in result

