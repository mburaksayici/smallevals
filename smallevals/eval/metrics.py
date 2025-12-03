"""
Retrieval metrics calculation and aggregation.

IMPORTANT: This is the SINGLE SOURCE OF TRUTH for all metric calculations.
Metrics are calculated here during evaluation and saved to evaluation_metrics.json.
The UI and reports load these pre-calculated metrics - they never recalculate.

This ensures consistency across:
- evaluation_metrics.json (saved during evaluation)
- report.html (generated during evaluation)
- UI dashboard (displays saved metrics)
"""

from typing import List, Dict, Any, Optional
import statistics
import warnings

from smallevals.utils.logger import logger


def precision_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
    """
    Calculate Precision@K.
    
    .. deprecated::
        Precision@K is partially deprecated. Consider using nDCG@K instead,
        which provides better ranking quality assessment with position discounting.

    Args:
        retrieved: List of retrieved items (top-k)
        relevant: List of relevant items
        k: Value of K

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    warnings.warn(
        "precision_at_k is partially deprecated. Consider using ndcg_at_k instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if k == 0:
        return 0.0

    relevant_set = set(relevant or [])
    
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for item in top_k if item in relevant_set)

    return relevant_retrieved / k


def recall_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
    """
    Calculate Recall@K.

    Args:
        retrieved: List of retrieved items (top-k)
        relevant: List of relevant items
        k: Value of K

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0

    top_k = retrieved[:k] if retrieved else []
    relevant_set = set(relevant)
    retrieved_set = set(top_k)

    relevant_retrieved = len(relevant_set & retrieved_set)
    return relevant_retrieved / len(relevant_set) if relevant_set else 0.0


def mean_reciprocal_rank(retrieved: List[Any], relevant: List[Any]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        retrieved: List of retrieved items
        relevant: List of relevant items

    Returns:
        MRR score (0.0 to 1.0)
    """
    if not relevant or not retrieved:
        return 0.0

    relevant_set = set(relevant)

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
    """
    Calculate Hit Rate@K (whether at least one relevant item is in top-k).

    Args:
        retrieved: List of retrieved items (top-k)
        relevant: List of relevant items
        k: Value of K

    Returns:
        Hit Rate@K score (0.0 or 1.0)
    """
    if not relevant or not retrieved:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)
    retrieved_set = set(top_k)

    return 1.0 if (relevant_set & retrieved_set) else 0.0


def ndcg_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@K).
    
    nDCG provides a ranking quality metric that gives higher weight to relevant
    items appearing at the top of the retrieval list. It uses logarithmic position
    discounting to penalize relevant items that appear lower in the ranking.
    
    DCG@K = sum(relevance_i / log2(position_i + 1)) for i in [1, k]
    nDCG@K = DCG@K / IDCG@K (ideal DCG with all relevant items at top)

    Args:
        retrieved: List of retrieved items (top-k)
        relevant: List of relevant items
        k: Value of K

    Returns:
        nDCG@K score (0.0 to 1.0)
    """
    if not relevant or not retrieved or k == 0:
        return 0.0
    
    import math
    
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    
    # Calculate DCG@K
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant_set:
            # Binary relevance: 1 if relevant, 0 otherwise
            # Discount by log2(position + 1)
            dcg += 1.0 / math.log2(i + 1)
    
    # Calculate Ideal DCG (IDCG) - all relevant items at top positions
    # For binary relevance, IDCG is sum of 1/log2(i+1) for min(k, num_relevant) positions
    num_relevant = len(relevant_set)
    idcg = 0.0
    for i in range(1, min(k, num_relevant) + 1):
        idcg += 1.0 / math.log2(i + 1)
    
    # Normalize
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def calculate_retrieval_metrics(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_chunk: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Calculate all retrieval metrics for a single query using VDB's ID.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries with "id" field
        relevant_chunk: Relevant chunk dictionary with "id" field
        top_k: Value of K for metrics

    Returns:
        Dictionary with metric scores
    """
    # Extract VDB's IDs - simple and direct
    retrieved_ids = [chunk.get("id", "") for chunk in retrieved_chunks]
    relevant_id = relevant_chunk.get("id", "")
    
    if not relevant_id:
        logger.warning("No ID found in relevant_chunk")
        return {
            f"precision@{top_k}": 0.0,
            f"recall@{top_k}": 0.0,
            "mrr": 0.0,
            f"hit_rate@{top_k}": 0.0,
            f"ndcg@{top_k}": 0.0,
        }
    
    # Use VDB's ID matching
    relevant_list = [relevant_id]
    
    # Calculate metrics (suppress deprecation warning for precision_at_k)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        precision_score = precision_at_k(retrieved_ids, relevant_list, top_k)
    
    metrics = {
        f"precision@{top_k}": precision_score,
        f"recall@{top_k}": recall_at_k(retrieved_ids, relevant_list, top_k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_list),
        f"hit_rate@{top_k}": hit_rate_at_k(retrieved_ids, relevant_list, top_k),
        f"ndcg@{top_k}": ndcg_at_k(retrieved_ids, relevant_list, top_k),
    }

    return metrics


def aggregate_metrics(
    per_sample_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples.

    Args:
        per_sample_metrics: List of metric dictionaries from each sample

    Returns:
        Dictionary with averaged metric scores
    """
    if not per_sample_metrics:
        return {}

    # Collect all metric names
    all_keys = set()
    for metrics in per_sample_metrics:
        all_keys.update(metrics.keys())

    # Calculate averages
    aggregated = {}
    for key in all_keys:
        values = [metrics.get(key, 0.0) for metrics in per_sample_metrics if key in metrics]
        if values:
            aggregated[key] = statistics.mean(values)
        else:
            aggregated[key] = 0.0

    return aggregated


def calculate_retrieval_metrics_full(
    qa_dataset: List[Dict[str, Any]],
    retrieval_results: List[List[Dict[str, Any]]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Calculate retrieval metrics for entire dataset using VDB's ID.

    Args:
        qa_dataset: List of Q/A pairs with "id" field (VDB's ID)
        retrieval_results: List of retrieval results for each Q/A pair
        top_k: Value of K for metrics

    Returns:
        Dictionary with aggregated and per-sample metrics
    """
    per_sample_metrics = []
    
    for qa_pair, retrieved in zip(qa_dataset, retrieval_results):
        # Use VDB's ID from qa_pair
        relevant_chunk = {"id": qa_pair.get("id", "")}
        
        # Keep passage for reference (but not used for matching)
        if "passage" in qa_pair:
            relevant_chunk["text"] = qa_pair["passage"]
        
        # Skip if no ID
        if not relevant_chunk["id"]:
            logger.warning("QA pair missing id, skipping")
            continue

        sample_metrics = calculate_retrieval_metrics(
            retrieved, relevant_chunk, top_k=top_k
        )
        per_sample_metrics.append(sample_metrics)

    aggregated = aggregate_metrics(per_sample_metrics)

    return {
        "aggregated": aggregated,
        "per_sample": per_sample_metrics,
    }

