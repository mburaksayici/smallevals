"""Unit tests for metrics calculation."""

import pytest
from evalvd.eval.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    calculate_retrieval_metrics,
    aggregate_metrics,
    calculate_retrieval_metrics_full,
)


class TestBasicMetrics:
    """Test basic metric functions."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        retrieved = ["id1", "id2", "id3", "id4", "id5"]
        relevant = ["id1", "id3", "id6"]
        
        # Precision@5: 2 relevant out of 5 retrieved = 0.4
        assert precision_at_k(retrieved, relevant, k=5) == 0.4
        # Precision@3: 2 relevant out of 3 retrieved = 0.666...
        assert abs(precision_at_k(retrieved, relevant, k=3) - 2/3) < 0.001
        # Precision@1: 1 relevant out of 1 retrieved = 1.0
        assert precision_at_k(retrieved, relevant, k=1) == 1.0
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        retrieved = ["id1", "id2", "id3", "id4", "id5"]
        relevant = ["id1", "id3", "id6"]
        
        # Recall@5: 2 relevant retrieved out of 3 total relevant = 2/3
        assert abs(recall_at_k(retrieved, relevant, k=5) - 2/3) < 0.001
        # Recall@1: 1 relevant retrieved out of 3 total relevant = 1/3
        assert abs(recall_at_k(retrieved, relevant, k=1) - 1/3) < 0.001
    
    def test_mean_reciprocal_rank(self):
        """Test MRR calculation."""
        retrieved = ["id1", "id2", "id3"]
        relevant = ["id2", "id4"]
        
        # First relevant at rank 2, so MRR = 1/2 = 0.5
        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 0.5
        
        # No relevant items
        assert mean_reciprocal_rank(["id1", "id2"], ["id3"]) == 0.0
        
        # First item is relevant
        assert mean_reciprocal_rank(["id1", "id2"], ["id1"]) == 1.0
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        retrieved = ["id1", "id2", "id3"]
        relevant = ["id2", "id4"]
        
        # At least one relevant in top-k
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
        assert hit_rate_at_k(retrieved, relevant, k=2) == 1.0
        assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
        
        # No relevant items
        assert hit_rate_at_k(["id1", "id2"], ["id3"], k=2) == 0.0


class TestRetrievalMetrics:
    """Test retrieval metrics calculation with chunk IDs."""
    
    def test_calculate_retrieval_metrics_with_chunk_ids(self):
        """Test metrics calculation using chunk IDs."""
        retrieved_chunks = [
            {"id": "chunk_1", "text": "text1", "score": 0.9},
            {"id": "chunk_2", "text": "text2", "score": 0.8},
            {"id": "chunk_3", "text": "text3", "score": 0.7},
        ]
        relevant_chunk = {"chunk_id": "chunk_1", "text": "text1"}
        
        metrics = calculate_retrieval_metrics(retrieved_chunks, relevant_chunk, top_k=3)
        
        assert "precision@3" in metrics
        assert "recall@3" in metrics
        assert "mrr" in metrics
        assert "hit_rate@3" in metrics
        
        # chunk_1 is first, so precision@3 = 1/3, recall@3 = 1/1, mrr = 1/1 = 1.0
        assert abs(metrics["precision@3"] - 1/3) < 0.001
        assert metrics["recall@3"] == 1.0
        assert metrics["mrr"] == 1.0
        assert metrics["hit_rate@3"] == 1.0
    
    def test_calculate_retrieval_metrics_with_metadata_chunk_id(self):
        """Test metrics calculation with chunk_id in metadata."""
        retrieved_chunks = [
            {"id": "chunk_1", "text": "text1", "metadata": {"chunk_id": "chunk_1"}},
            {"id": "chunk_2", "text": "text2", "metadata": {"chunk_id": "chunk_2"}},
        ]
        relevant_chunk = {"chunk_id": "chunk_2"}
        
        metrics = calculate_retrieval_metrics(retrieved_chunks, relevant_chunk, top_k=2)
        
        # chunk_2 is second, so precision@2 = 1/2, mrr = 1/2
        assert metrics["precision@2"] == 0.5
        assert metrics["mrr"] == 0.5
        assert metrics["hit_rate@2"] == 1.0
    
    def test_calculate_retrieval_metrics_no_match(self):
        """Test metrics when no chunks match."""
        retrieved_chunks = [
            {"id": "chunk_1", "text": "text1"},
            {"id": "chunk_2", "text": "text2"},
        ]
        relevant_chunk = {"chunk_id": "chunk_3"}
        
        metrics = calculate_retrieval_metrics(retrieved_chunks, relevant_chunk, top_k=2)
        
        assert metrics["precision@2"] == 0.0
        assert metrics["recall@2"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["hit_rate@2"] == 0.0


class TestAggregation:
    """Test metric aggregation."""
    
    def test_aggregate_metrics(self):
        """Test aggregating metrics across samples."""
        per_sample = [
            {"precision@5": 0.8, "recall@5": 0.6, "mrr": 0.5},
            {"precision@5": 0.9, "recall@5": 0.7, "mrr": 0.6},
            {"precision@5": 0.7, "recall@5": 0.5, "mrr": 0.4},
        ]
        
        aggregated = aggregate_metrics(per_sample)
        
        assert "precision@5" in aggregated
        assert "recall@5" in aggregated
        assert "mrr" in aggregated
        
        # Average of [0.8, 0.9, 0.7] = 0.8
        assert abs(aggregated["precision@5"] - 0.8) < 0.001
        # Average of [0.6, 0.7, 0.5] = 0.6
        assert abs(aggregated["recall@5"] - 0.6) < 0.001
        # Average of [0.5, 0.6, 0.4] = 0.5
        assert abs(aggregated["mrr"] - 0.5) < 0.001
    
    def test_aggregate_metrics_empty(self):
        """Test aggregation with empty list."""
        assert aggregate_metrics([]) == {}


class TestFullMetrics:
    """Test full metrics calculation."""
    
    def test_calculate_retrieval_metrics_full(self):
        """Test full metrics calculation for dataset."""
        qa_dataset = [
            {"question": "q1", "answer": "a1", "chunk_id": "chunk_1", "passage": "text1"},
            {"question": "q2", "answer": "a2", "chunk_id": "chunk_2", "passage": "text2"},
        ]
        retrieval_results = [
            [{"id": "chunk_1", "text": "text1"}],  # First query: correct
            [{"id": "chunk_3", "text": "text3"}],  # Second query: wrong
        ]
        
        result = calculate_retrieval_metrics_full(qa_dataset, retrieval_results, top_k=1)
        
        assert "aggregated" in result
        assert "per_sample" in result
        assert len(result["per_sample"]) == 2
        
        # First query: precision@1 = 1.0, mrr = 1.0
        # Second query: precision@1 = 0.0, mrr = 0.0
        # Average precision@1 = 0.5
        assert abs(result["aggregated"]["precision@1"] - 0.5) < 0.001
        assert abs(result["aggregated"]["mrr"] - 0.5) < 0.001

