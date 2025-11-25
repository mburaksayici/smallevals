"""Unit tests for QA generator."""

import pytest
from unittest.mock import Mock, patch

from evalvd.generation.qa_generator import QAGenerator
from evalvd.exceptions import ValidationError


def test_qa_generator_validation():
    """Test input validation for QAGenerator."""
    # Test invalid batch_size
    with pytest.raises(ValidationError):
        QAGenerator(batch_size=0)
    
    with pytest.raises(ValidationError):
        QAGenerator(batch_size=-1)


def test_qa_generator_generate_from_chunks_validation():
    """Test validation in generate_from_chunks."""
    # Mock model loader to avoid actual model loading
    with patch('evalvd.generation.qa_generator.ModelLoader'):
        generator = QAGenerator(batch_size=2)
        
        # Test empty chunks
        with pytest.raises(ValidationError):
            generator.generate_from_chunks([])
        
        # Test invalid chunks (no text field)
        with pytest.raises(ValidationError):
            generator.generate_from_chunks([{"no_text": "value"}])


def test_qa_generator_format_prompt():
    """Test prompt formatting."""
    with patch('evalvd.generation.qa_generator.ModelLoader'):
        generator = QAGenerator()
        
        passage = "This is a test passage."
        prompt = generator.format_prompt(passage)
        
        assert passage in prompt
        assert "question" in prompt.lower() or "answer" in prompt.lower()


def test_qa_generator_generate_qa_batch_validation():
    """Test validation in generate_qa_batch."""
    with patch('evalvd.generation.qa_generator.ModelLoader'):
        generator = QAGenerator(batch_size=2)
        
        # Test invalid max_retries
        with pytest.raises(ValidationError):
            generator.generate_qa_batch(["passage1"], max_retries=-1)

