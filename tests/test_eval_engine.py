"""Integration tests for evaluation engine."""

import pytest
import json
from pathlib import Path
import tempfile

from smallevals.eval.engine import generate_qa_from_vectordb, evaluate_retrievals
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

