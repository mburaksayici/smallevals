"""Retrieval metrics calculation and aggregation."""

from typing import List, Dict, Any, Optional
import statistics
import warnings

from smallevals.utils.logger import logger


def precision_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
    """
    Calculate Precision@K.

    Args:
        retrieved: List of retrieved items (top-k)
        relevant: List of relevant items
        k: Value of K

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if not retrieved or k == 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant) if relevant else set()
    
    if not relevant_set:
        return 0.0

    relevant_retrieved = sum(1 for item in top_k if item in relevant_set)
    return relevant_retrieved / min(len(top_k), k)


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


def _extract_chunk_id(chunk: Dict[str, Any]) -> Optional[str]:
    """
    Extract chunk ID from a chunk dictionary.
    
    Args:
        chunk: Chunk dictionary
        
    Returns:
        Chunk ID string or None if not found
    """
    if not isinstance(chunk, dict):
        return None
    
    # Try "id" field first
    if "id" in chunk and chunk["id"]:
        return str(chunk["id"])
    
    # Try metadata["chunk_id"]
    if "metadata" in chunk and isinstance(chunk["metadata"], dict):
        if "chunk_id" in chunk["metadata"]:
            return str(chunk["metadata"]["chunk_id"])
    
    return None


def calculate_retrieval_metrics(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_chunk: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Calculate all retrieval metrics for a single query using chunk ID matching.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        relevant_chunk: Relevant chunk dictionary (ground truth) with "chunk_id" or "id"
        top_k: Value of K for metrics

    Returns:
        Dictionary with metric scores
    """
    # Extract chunk IDs from retrieved chunks
    retrieved_ids = []
    for chunk in retrieved_chunks:
        chunk_id = _extract_chunk_id(chunk)
        if chunk_id:
            retrieved_ids.append(chunk_id)
        else:
            logger.warning("Retrieved chunk missing ID, skipping from metrics")
    
    # Extract relevant chunk ID
    relevant_id = _extract_chunk_id(relevant_chunk)
    
    # Fallback to text matching if IDs not available (with warning)
    if not relevant_id:
        # Try to get from qa_pair structure (chunk_id field)
        if "chunk_id" in relevant_chunk:
            relevant_id = str(relevant_chunk["chunk_id"])
        else:
            # Fallback to text matching
            warnings.warn(
                "No chunk_id found in relevant_chunk, falling back to text matching. "
                "This is deprecated - ensure chunk_id is provided for accurate metrics.",
                DeprecationWarning,
                stacklevel=2
            )
            retrieved_texts = [chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in retrieved_chunks]
            relevant_text = relevant_chunk.get("text", "") if isinstance(relevant_chunk, dict) else str(relevant_chunk)
            relevant_list = [relevant_text] if relevant_text else []
            
            return {
                f"precision@{top_k}": precision_at_k(retrieved_texts, relevant_list, top_k),
                f"recall@{top_k}": recall_at_k(retrieved_texts, relevant_list, top_k),
                "mrr": mean_reciprocal_rank(retrieved_texts, relevant_list),
                f"hit_rate@{top_k}": hit_rate_at_k(retrieved_texts, relevant_list, top_k),
            }
    
    # Use chunk ID matching
    relevant_list = [relevant_id] if relevant_id else []
    
    # Calculate metrics
    metrics = {
        f"precision@{top_k}": precision_at_k(retrieved_ids, relevant_list, top_k),
        f"recall@{top_k}": recall_at_k(retrieved_ids, relevant_list, top_k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_list),
        f"hit_rate@{top_k}": hit_rate_at_k(retrieved_ids, relevant_list, top_k),
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
    Calculate retrieval metrics for entire dataset using chunk ID matching.

    Args:
        qa_dataset: List of Q/A pairs with "chunk_id" field (ground truth) or "passage" for backward compatibility
        retrieval_results: List of retrieval results for each Q/A pair
        top_k: Value of K for metrics

    Returns:
        Dictionary with aggregated and per-sample metrics
    """
    per_sample_metrics = []

    for qa_pair, retrieved in zip(qa_dataset, retrieval_results):
        # Build relevant_chunk dict with chunk_id if available
        relevant_chunk = {}
        
        # Prefer chunk_id from qa_pair
        if "chunk_id" in qa_pair:
            relevant_chunk["chunk_id"] = qa_pair["chunk_id"]
        # Also include id field for compatibility
        if "chunk_id" in qa_pair:
            relevant_chunk["id"] = qa_pair["chunk_id"]
        
        # Include passage for backward compatibility (fallback to text matching)
        if "passage" in qa_pair:
            relevant_chunk["text"] = qa_pair["passage"]
        
        # Include metadata if present
        if "metadata" in qa_pair:
            relevant_chunk["metadata"] = qa_pair["metadata"]
        
        # Skip if no identifying information
        if not relevant_chunk:
            logger.warning("QA pair missing chunk_id or passage, skipping")
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

