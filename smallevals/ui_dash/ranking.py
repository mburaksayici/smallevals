"""
Data manipulation utilities for retrieval evaluation results.

IMPORTANT: This module ONLY manipulates DataFrames (filtering, sorting, slicing).
It does NOT calculate metrics. Metrics are calculated once during evaluation 
(in smallevals.eval.metrics) and saved to evaluation_metrics.json.

The UI dashboard loads and displays these pre-calculated metrics to ensure consistency.

Functions:
- filter_by_rank: Filter dataframe by specific rank
- get_rank_distribution: Count occurrences at each rank
- rank_by_metric: Sort dataframe by a column
"""

import pandas as pd
from typing import Dict, Any, Optional


def filter_by_rank(df: pd.DataFrame, rank: int) -> pd.DataFrame:
    """
    Filter dataframe to only include rows where chunk was found at a specific rank.
    
    Args:
        df: Results dataframe with 'chunk_position' column
        rank: Rank to filter by (e.g., 1 for rank 1)
        
    Returns:
        Filtered dataframe
    """
    if 'chunk_position' not in df.columns:
        return pd.DataFrame()
    
    return df[df['chunk_position'] == rank].copy()


def get_rank_distribution(df: pd.DataFrame, top_k: int = 5) -> Dict[str, int]:
    """
    Get distribution of retrieval ranks.
    
    Args:
        df: Results dataframe with 'chunk_position' column
        top_k: Maximum rank to consider
        
    Returns:
        Dictionary with keys like 'rank_1', 'rank_2', ..., 'rank_{top_k}', 'not_found', 'total'
    """
    if df.empty or 'chunk_position' not in df.columns:
        return {'total': 0}
    
    distribution = {}
    
    # Count occurrences at each rank
    for rank in range(1, top_k + 1):
        count = len(df[df['chunk_position'] == rank])
        distribution[f'rank_{rank}'] = count
    
    # Count not found (NaN or > top_k)
    not_found = len(df[df['chunk_position'].isna() | (df['chunk_position'] > top_k)])
    distribution['not_found'] = not_found
    
    distribution['total'] = len(df)
    
    return distribution


def rank_by_metric(df: pd.DataFrame, metric: str = "position", ascending: bool = True) -> pd.DataFrame:
    """
    Sort dataframe by a metric column.
    
    Args:
        df: Results dataframe
        metric: Metric to sort by ('position', 'mrr', 'hit_rate', etc.)
        ascending: Sort ascending (True) or descending (False)
        
    Returns:
        Sorted dataframe
    """
    if df.empty:
        return df.copy()
    
    # Map metric names to column names
    metric_map = {
        'position': 'chunk_position',
        'mrr': 'mrr',
        'hit_rate': 'hit_rate',
    }
    
    sort_col = metric_map.get(metric, metric)
    
    # If column doesn't exist, return original dataframe
    if sort_col not in df.columns:
        return df.copy()
    
    # Handle NaN values - put them at the end
    df_sorted = df.copy()
    df_sorted['_sort_key'] = df_sorted[sort_col].fillna(float('inf') if ascending else float('-inf'))
    
    return df_sorted.sort_values('_sort_key', ascending=ascending).drop(columns=['_sort_key'])