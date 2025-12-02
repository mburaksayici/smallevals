"""Comprehensive unit tests for metrics calculation using fake documents."""

import pytest
from smallevals.eval.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    calculate_retrieval_metrics,
    aggregate_metrics,
    calculate_retrieval_metrics_full,
)


def test_all_metrics_comprehensive():
    """
    Comprehensive test of all metrics functions using 10 fake documents.
    Tests both perfect scenario (10/10 correct) and worst case (0/10 correct).
    """
    
    # ===== Setup Test Data =====
    
    # 10 relevant document IDs
    relevant_ids = [f"chunk_{i}" for i in range(10)]
    
    # Perfect case: retrieved IDs match relevant IDs exactly
    perfect_retrieved_ids = [f"chunk_{i}" for i in range(10)]
    
    # Worst case: retrieved IDs don't match any relevant IDs
    worst_retrieved_ids = [f"wrong_{i}" for i in range(10)]
    
    # Partial case: 5 out of 10 correct
    partial_retrieved_ids = [f"chunk_{i}" for i in range(5)] + [f"wrong_{i}" for i in range(5)]
    
    # ===== PERFECT SCENARIO (10/10 correct) =====
    print("\n=== Testing Perfect Scenario (10/10 correct) ===")
    
    # Test precision_at_k - Perfect case
    assert precision_at_k(perfect_retrieved_ids, relevant_ids, k=10) == 1.0, \
        "Precision@10 should be 1.0 for perfect retrieval"
    assert precision_at_k(perfect_retrieved_ids, relevant_ids, k=5) == 1.0, \
        "Precision@5 should be 1.0 for perfect retrieval"
    assert precision_at_k(perfect_retrieved_ids, relevant_ids, k=1) == 1.0, \
        "Precision@1 should be 1.0 for perfect retrieval"
    print("✓ precision_at_k: Perfect scenario passed")
    
    # Test recall_at_k - Perfect case
    assert recall_at_k(perfect_retrieved_ids, relevant_ids, k=10) == 1.0, \
        "Recall@10 should be 1.0 for perfect retrieval"
    assert recall_at_k(perfect_retrieved_ids, relevant_ids, k=5) == 0.5, \
        "Recall@5 should be 0.5 (5 out of 10 relevant retrieved)"
    assert recall_at_k(perfect_retrieved_ids, relevant_ids, k=1) == 0.1, \
        "Recall@1 should be 0.1 (1 out of 10 relevant retrieved)"
    print("✓ recall_at_k: Perfect scenario passed")
    
    # Test mean_reciprocal_rank - Perfect case
    assert mean_reciprocal_rank(perfect_retrieved_ids, relevant_ids) == 1.0, \
        "MRR should be 1.0 when first item is relevant"
    print("✓ mean_reciprocal_rank: Perfect scenario passed")
    
    # Test hit_rate_at_k - Perfect case
    assert hit_rate_at_k(perfect_retrieved_ids, relevant_ids, k=10) == 1.0, \
        "Hit rate@10 should be 1.0 for perfect retrieval"
    assert hit_rate_at_k(perfect_retrieved_ids, relevant_ids, k=5) == 1.0, \
        "Hit rate@5 should be 1.0 for perfect retrieval"
    assert hit_rate_at_k(perfect_retrieved_ids, relevant_ids, k=1) == 1.0, \
        "Hit rate@1 should be 1.0 for perfect retrieval"
    print("✓ hit_rate_at_k: Perfect scenario passed")
    
    # ===== WORST CASE (0/10 correct) =====
    print("\n=== Testing Worst Case (0/10 correct) ===")
    
    # Test precision_at_k - Worst case
    assert precision_at_k(worst_retrieved_ids, relevant_ids, k=10) == 0.0, \
        "Precision@10 should be 0.0 when no relevant docs retrieved"
    assert precision_at_k(worst_retrieved_ids, relevant_ids, k=5) == 0.0, \
        "Precision@5 should be 0.0 when no relevant docs retrieved"
    assert precision_at_k(worst_retrieved_ids, relevant_ids, k=1) == 0.0, \
        "Precision@1 should be 0.0 when no relevant docs retrieved"
    print("✓ precision_at_k: Worst case passed")
    
    # Test recall_at_k - Worst case
    assert recall_at_k(worst_retrieved_ids, relevant_ids, k=10) == 0.0, \
        "Recall@10 should be 0.0 when no relevant docs retrieved"
    assert recall_at_k(worst_retrieved_ids, relevant_ids, k=5) == 0.0, \
        "Recall@5 should be 0.0 when no relevant docs retrieved"
    assert recall_at_k(worst_retrieved_ids, relevant_ids, k=1) == 0.0, \
        "Recall@1 should be 0.0 when no relevant docs retrieved"
    print("✓ recall_at_k: Worst case passed")
    
    # Test mean_reciprocal_rank - Worst case
    assert mean_reciprocal_rank(worst_retrieved_ids, relevant_ids) == 0.0, \
        "MRR should be 0.0 when no relevant docs retrieved"
    print("✓ mean_reciprocal_rank: Worst case passed")
    
    # Test hit_rate_at_k - Worst case
    assert hit_rate_at_k(worst_retrieved_ids, relevant_ids, k=10) == 0.0, \
        "Hit rate@10 should be 0.0 when no relevant docs retrieved"
    assert hit_rate_at_k(worst_retrieved_ids, relevant_ids, k=5) == 0.0, \
        "Hit rate@5 should be 0.0 when no relevant docs retrieved"
    assert hit_rate_at_k(worst_retrieved_ids, relevant_ids, k=1) == 0.0, \
        "Hit rate@1 should be 0.0 when no relevant docs retrieved"
    print("✓ hit_rate_at_k: Worst case passed")
    
    # ===== PARTIAL CASE (5/10 correct) =====
    print("\n=== Testing Partial Case (5/10 correct) ===")
    
    # Test precision_at_k - Partial case
    assert precision_at_k(partial_retrieved_ids, relevant_ids, k=10) == 0.5, \
        "Precision@10 should be 0.5 (5 relevant out of 10 retrieved)"
    assert precision_at_k(partial_retrieved_ids, relevant_ids, k=5) == 1.0, \
        "Precision@5 should be 1.0 (5 relevant out of 5 retrieved)"
    print("✓ precision_at_k: Partial case passed")
    
    # Test recall_at_k - Partial case
    assert recall_at_k(partial_retrieved_ids, relevant_ids, k=10) == 0.5, \
        "Recall@10 should be 0.5 (5 out of 10 relevant retrieved)"
    assert recall_at_k(partial_retrieved_ids, relevant_ids, k=5) == 0.5, \
        "Recall@5 should be 0.5 (5 out of 10 relevant retrieved)"
    print("✓ recall_at_k: Partial case passed")
    
    # ===== Test calculate_retrieval_metrics =====
    print("\n=== Testing calculate_retrieval_metrics ===")
    
    # Perfect case with chunk dictionaries
    perfect_chunks = [{"id": f"chunk_{i}", "text": f"text_{i}"} for i in range(10)]
    relevant_chunk_perfect = {"chunk_id": "chunk_0", "text": "text_0"}
    
    metrics_perfect = calculate_retrieval_metrics(perfect_chunks, relevant_chunk_perfect, top_k=10)
    assert metrics_perfect["precision@10"] == 0.1, "Precision@10 should be 0.1 (1 relevant out of 10)"
    assert metrics_perfect["recall@10"] == 1.0, "Recall@10 should be 1.0 (1 out of 1 relevant retrieved)"
    assert metrics_perfect["mrr"] == 1.0, "MRR should be 1.0 (first item is relevant)"
    assert metrics_perfect["hit_rate@10"] == 1.0, "Hit rate@10 should be 1.0"
    print("✓ calculate_retrieval_metrics: Perfect case passed")
    
    # Worst case with chunk dictionaries
    worst_chunks = [{"id": f"wrong_{i}", "text": f"text_{i}"} for i in range(10)]
    relevant_chunk_worst = {"chunk_id": "chunk_0", "text": "text_0"}
    
    metrics_worst = calculate_retrieval_metrics(worst_chunks, relevant_chunk_worst, top_k=10)
    assert metrics_worst["precision@10"] == 0.0, "Precision@10 should be 0.0"
    assert metrics_worst["recall@10"] == 0.0, "Recall@10 should be 0.0"
    assert metrics_worst["mrr"] == 0.0, "MRR should be 0.0"
    assert metrics_worst["hit_rate@10"] == 0.0, "Hit rate@10 should be 0.0"
    print("✓ calculate_retrieval_metrics: Worst case passed")
    
    # ===== Test aggregate_metrics =====
    print("\n=== Testing aggregate_metrics ===")
    
    # Perfect aggregation (all 1.0)
    perfect_samples = [
        {"precision@5": 1.0, "recall@5": 1.0, "mrr": 1.0, "hit_rate@5": 1.0}
        for _ in range(10)
    ]
    agg_perfect = aggregate_metrics(perfect_samples)
    assert agg_perfect["precision@5"] == 1.0, "Aggregated precision should be 1.0"
    assert agg_perfect["recall@5"] == 1.0, "Aggregated recall should be 1.0"
    assert agg_perfect["mrr"] == 1.0, "Aggregated MRR should be 1.0"
    assert agg_perfect["hit_rate@5"] == 1.0, "Aggregated hit rate should be 1.0"
    print("✓ aggregate_metrics: Perfect aggregation passed")
    
    # Worst aggregation (all 0.0)
    worst_samples = [
        {"precision@5": 0.0, "recall@5": 0.0, "mrr": 0.0, "hit_rate@5": 0.0}
        for _ in range(10)
    ]
    agg_worst = aggregate_metrics(worst_samples)
    assert agg_worst["precision@5"] == 0.0, "Aggregated precision should be 0.0"
    assert agg_worst["recall@5"] == 0.0, "Aggregated recall should be 0.0"
    assert agg_worst["mrr"] == 0.0, "Aggregated MRR should be 0.0"
    assert agg_worst["hit_rate@5"] == 0.0, "Aggregated hit rate should be 0.0"
    print("✓ aggregate_metrics: Worst aggregation passed")
    
    # Mixed aggregation (should average to 0.5)
    mixed_samples = perfect_samples + worst_samples
    agg_mixed = aggregate_metrics(mixed_samples)
    assert abs(agg_mixed["precision@5"] - 0.5) < 0.01, "Aggregated precision should be ~0.5"
    assert abs(agg_mixed["recall@5"] - 0.5) < 0.01, "Aggregated recall should be ~0.5"
    assert abs(agg_mixed["mrr"] - 0.5) < 0.01, "Aggregated MRR should be ~0.5"
    assert abs(agg_mixed["hit_rate@5"] - 0.5) < 0.01, "Aggregated hit rate should be ~0.5"
    print("✓ aggregate_metrics: Mixed aggregation passed")
    
    # ===== Test calculate_retrieval_metrics_full =====
    print("\n=== Testing calculate_retrieval_metrics_full ===")
    
    # Perfect case: all queries retrieve correct chunks
    qa_dataset_perfect = [
        {"question": f"q{i}", "answer": f"a{i}", "chunk_id": f"chunk_{i}"}
        for i in range(10)
    ]
    retrieval_results_perfect = [
        [{"id": f"chunk_{i}", "text": f"text_{i}"}]
        for i in range(10)
    ]
    
    full_metrics_perfect = calculate_retrieval_metrics_full(
        qa_dataset_perfect, retrieval_results_perfect, top_k=1
    )
    assert "aggregated" in full_metrics_perfect
    assert "per_sample" in full_metrics_perfect
    assert len(full_metrics_perfect["per_sample"]) == 10
    assert full_metrics_perfect["aggregated"]["precision@1"] == 1.0, \
        "Average precision@1 should be 1.0 for perfect retrieval"
    assert full_metrics_perfect["aggregated"]["recall@1"] == 1.0, \
        "Average recall@1 should be 1.0 for perfect retrieval"
    assert full_metrics_perfect["aggregated"]["mrr"] == 1.0, \
        "Average MRR should be 1.0 for perfect retrieval"
    print("✓ calculate_retrieval_metrics_full: Perfect case passed")
    
    # Worst case: all queries retrieve wrong chunks
    qa_dataset_worst = [
        {"question": f"q{i}", "answer": f"a{i}", "chunk_id": f"chunk_{i}"}
        for i in range(10)
    ]
    retrieval_results_worst = [
        [{"id": f"wrong_{i}", "text": f"text_{i}"}]
        for i in range(10)
    ]
    
    full_metrics_worst = calculate_retrieval_metrics_full(
        qa_dataset_worst, retrieval_results_worst, top_k=1
    )
    assert full_metrics_worst["aggregated"]["precision@1"] == 0.0, \
        "Average precision@1 should be 0.0 for worst case"
    assert full_metrics_worst["aggregated"]["recall@1"] == 0.0, \
        "Average recall@1 should be 0.0 for worst case"
    assert full_metrics_worst["aggregated"]["mrr"] == 0.0, \
        "Average MRR should be 0.0 for worst case"
    print("✓ calculate_retrieval_metrics_full: Worst case passed")
    
    # Mixed case: 5 correct, 5 wrong
    qa_dataset_mixed = [
        {"question": f"q{i}", "answer": f"a{i}", "chunk_id": f"chunk_{i}"}
        for i in range(10)
    ]
    retrieval_results_mixed = (
        [{"id": f"chunk_{i}", "text": f"text_{i}"}] for i in range(5)
    )
    retrieval_results_mixed = list(retrieval_results_mixed) + [
        [{"id": f"wrong_{i}", "text": f"text_{i}"}] for i in range(5)
    ]
    
    full_metrics_mixed = calculate_retrieval_metrics_full(
        qa_dataset_mixed, retrieval_results_mixed, top_k=1
    )
    assert abs(full_metrics_mixed["aggregated"]["precision@1"] - 0.5) < 0.01, \
        "Average precision@1 should be ~0.5 for mixed case"
    assert abs(full_metrics_mixed["aggregated"]["mrr"] - 0.5) < 0.01, \
        "Average MRR should be ~0.5 for mixed case"
    print("✓ calculate_retrieval_metrics_full: Mixed case passed")
    
    print("\n=== All Comprehensive Tests Passed ✓ ===")


def test_edge_cases():
    """Test edge cases for metrics functions."""
    
    # Empty lists
    assert precision_at_k([], ["id1"], k=5) == 0.0
    assert recall_at_k([], ["id1"], k=5) == 0.0
    assert mean_reciprocal_rank([], ["id1"]) == 0.0
    assert hit_rate_at_k([], ["id1"], k=5) == 0.0
    
    # Empty relevant list
    assert precision_at_k(["id1"], [], k=5) == 0.0
    assert recall_at_k(["id1"], [], k=5) == 0.0
    assert mean_reciprocal_rank(["id1"], []) == 0.0
    assert hit_rate_at_k(["id1"], [], k=5) == 0.0
    
    # k=0
    assert precision_at_k(["id1"], ["id1"], k=0) == 0.0
    
    # k larger than retrieved list
    retrieved = ["id1", "id2"]
    relevant = ["id1", "id3"]
    assert precision_at_k(retrieved, relevant, k=10) == 0.5
    
    # Relevant item at different positions
    retrieved_rank_2 = ["wrong1", "id1", "wrong2"]
    assert mean_reciprocal_rank(retrieved_rank_2, ["id1"]) == 0.5  # 1/2
    
    retrieved_rank_3 = ["wrong1", "wrong2", "id1"]
    assert abs(mean_reciprocal_rank(retrieved_rank_3, ["id1"]) - 1/3) < 0.01  # 1/3
    
    # Empty aggregate
    assert aggregate_metrics([]) == {}

    print("✓ Edge cases: All tests passed")


def test_specific_k_values():
    """Test metrics at different k values to ensure correctness."""
    
    # Retrieved: [chunk_0, chunk_1, chunk_2, chunk_3, chunk_4, wrong_0, wrong_1, wrong_2, wrong_3, wrong_4]
    # Relevant: [chunk_0, chunk_1, chunk_2, chunk_3, chunk_4, chunk_5, chunk_6, chunk_7, chunk_8, chunk_9]
    # So first 5 are correct, next 5 are wrong
    
    retrieved = [f"chunk_{i}" for i in range(5)] + [f"wrong_{i}" for i in range(5)]
    relevant = [f"chunk_{i}" for i in range(10)]
    
    # At k=1: 1 correct out of 1 retrieved, 1 out of 10 relevant
    assert precision_at_k(retrieved, relevant, k=1) == 1.0  # 1/1
    assert recall_at_k(retrieved, relevant, k=1) == 0.1  # 1/10
    
    # At k=5: 5 correct out of 5 retrieved, 5 out of 10 relevant
    assert precision_at_k(retrieved, relevant, k=5) == 1.0  # 5/5
    assert recall_at_k(retrieved, relevant, k=5) == 0.5  # 5/10
    
    # At k=10: 5 correct out of 10 retrieved, 5 out of 10 relevant
    assert precision_at_k(retrieved, relevant, k=10) == 0.5  # 5/10
    assert recall_at_k(retrieved, relevant, k=10) == 0.5  # 5/10
    
    # MRR should be 1.0 since first item is relevant
    assert mean_reciprocal_rank(retrieved, relevant) == 1.0
    
    print("✓ Specific k values: All tests passed")
