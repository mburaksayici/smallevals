"""RAG failure analysis - over-retrieved chunks."""

import pandas as pd
from typing import Dict, Tuple, List
import ast
import json
from collections import Counter


def _parse_retrieved_ids(retrieved_ids_str: str) -> List[str]:
    """Parse retrieved_ids string (can be JSON list or string representation)."""
    if pd.isna(retrieved_ids_str) or not retrieved_ids_str:
        return []
    
    try:
        # Try JSON parsing first
        if isinstance(retrieved_ids_str, str):
            # Try ast.literal_eval for Python list representation
            try:
                return ast.literal_eval(retrieved_ids_str)
            except (ValueError, SyntaxError):
                # Try JSON parsing
                try:
                    return json.loads(retrieved_ids_str)
                except json.JSONDecodeError:
                    return []
        return retrieved_ids_str if isinstance(retrieved_ids_str, list) else []
    except Exception:
        return []


def analyze_retrieval_frequency(
    df: pd.DataFrame,
    top_k: int = 5,
    threshold: float = 0.1
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze retrieval frequency - identify chunks that appear regardless of query.
    
    Args:
        df: DataFrame with 'retrieved_ids' column
        top_k: Top-k value used for retrieval
        threshold: Threshold for over-retrieval (fraction of queries where chunk appears)
        
    Returns:
        Tuple of (DataFrame with new columns, visualization data dict)
    """
    df = df.copy()
    
    # Count how many times each chunk appears in top-k across all queries
    chunk_frequencies = Counter()
    total_queries = len(df)
    
    for idx, row in df.iterrows():
        retrieved_ids_str = row.get('retrieved_ids', '[]')
        retrieved_ids = _parse_retrieved_ids(retrieved_ids_str)
        
        # Count appearances in top-k
        for chunk_id in retrieved_ids[:top_k]:
            chunk_frequencies[str(chunk_id)] += 1
    
    # Calculate frequency ratio (appearances / total queries)
    chunk_frequency_ratios = {
        chunk_id: count / total_queries
        for chunk_id, count in chunk_frequencies.items()
    }
    
    # Add columns to DataFrame
    df['retrieval_frequency'] = df['chunk_id'].astype(str).apply(
        lambda x: chunk_frequency_ratios.get(x, 0.0)
    )
    
    # Mark over-retrieved chunks
    df['is_over_retrieved'] = df['retrieval_frequency'] > threshold
    
    # Create visualization data
    over_retrieved_chunks = {
        chunk_id: {
            'frequency': ratio,
            'appearances': chunk_frequencies[chunk_id],
        }
        for chunk_id, ratio in chunk_frequency_ratios.items()
        if ratio > threshold
    }
    
    # Sort by frequency
    top_over_retrieved = sorted(
        over_retrieved_chunks.items(),
        key=lambda x: x[1]['frequency'],
        reverse=True
    )[:20]  # Top 20
    
    viz_data = {
        'over_retrieved_chunks': {
            chunk_id: stats
            for chunk_id, stats in top_over_retrieved
        },
        'total_over_retrieved': len(over_retrieved_chunks),
        'threshold': threshold,
    }
    
    return df, viz_data

