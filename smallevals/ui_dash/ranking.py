"""Ranking and metrics calculation utilities for retrieval evaluation results."""

import pandas as pd
from typing import Dict, Any, Optional
import numpy as np


def calculate_metrics_from_df(df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
    """
    Calculate retrieval metrics from a results dataframe.
    
    Args:
        df: DataFrame with columns including 'chunk_position' (rank where chunk was found)
        top_k: Value of K for metrics
        
    Returns:
        Dictionary with metrics: mrr, hit_rate@{top_k}, precision@{top_k}, recall@{top_k},
        num_queries, num_found, num_not_found
    """
    if df.empty:
        return {
            'mrr': 0.0,
            f'hit_rate@{top_k}': 0.0,
            f'precision@{top_k}': 0.0,
            f'recall@{top_k}': 0.0,
            'num_queries': 0,
            'num_found': 0,
            'num_not_found': 0,
        }
    
    # Ensure we have chunk_position column
    if 'chunk_position' not in df.columns:
        # If not present, assume all not found
        return {
            'mrr': 0.0,
            f'hit_rate@{top_k}': 0.0,
            f'precision@{top_k}': 0.0,
            f'recall@{top_k}': 0.0,
            'num_queries': len(df),
            'num_found': 0,
            'num_not_found': len(df),
        }
    
    # Filter out NaN positions (chunks not found)
    found_df = df[df['chunk_position'].notna()]
    
    # Calculate MRR (Mean Reciprocal Rank)
    if len(found_df) > 0:
        reciprocal_ranks = 1.0 / found_df['chunk_position']
        mrr = reciprocal_ranks.mean()
    else:
        mrr = 0.0
    
    # Count queries found in top-k
    found_in_topk = found_df[found_df['chunk_position'] <= top_k]
    num_found_in_topk = len(found_in_topk)
    num_queries = len(df)
    num_found = len(found_df)
    num_not_found = num_queries - num_found
    
    # Hit Rate@K: fraction of queries where relevant chunk found in top-k
    hit_rate = num_found_in_topk / num_queries if num_queries > 0 else 0.0
    
    # Precision@K: fraction of retrieved items that are relevant (assuming 1 relevant per query)
    # For single relevant item per query: precision@k = hit_rate@k
    precision = hit_rate
    
    # Recall@K: fraction of relevant items retrieved (assuming 1 relevant per query)
    # For single relevant item per query: recall@k = hit_rate@k
    recall = hit_rate
    
    return {
        'mrr': float(mrr),
        f'hit_rate@{top_k}': float(hit_rate),
        f'precision@{top_k}': float(precision),
        f'recall@{top_k}': float(recall),
        'num_queries': num_queries,
        'num_found': num_found_in_topk,  # Return count found in top-k (not total found)
        'num_not_found': num_not_found,
    }


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


def calculate_per_query_metrics(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Calculate per-query metrics and add them as columns to the dataframe.
    
    Args:
        df: Results dataframe with 'chunk_position' column
        top_k: Value of K for metrics
        
    Returns:
        DataFrame with additional metric columns: 'mrr', 'hit_rate', 'precision', 'recall'
    """
    result_df = df.copy()
    
    if 'chunk_position' not in df.columns:
        result_df['mrr'] = 0.0
        result_df['hit_rate'] = 0.0
        result_df['precision'] = 0.0
        result_df['recall'] = 0.0
        return result_df
    
    # Calculate MRR per query (reciprocal rank)
    def calc_mrr(position):
        if pd.isna(position):
            return 0.0
        return 1.0 / float(position)
    
    result_df['mrr'] = result_df['chunk_position'].apply(calc_mrr)
    
    # Calculate hit_rate per query (1 if found in top-k, 0 otherwise)
    def calc_hit_rate(position):
        if pd.isna(position):
            return 0.0
        return 1.0 if position <= top_k else 0.0
    
    result_df['hit_rate'] = result_df['chunk_position'].apply(calc_hit_rate)
    
    # For single relevant item per query, precision and recall equal hit_rate
    result_df['precision'] = result_df['hit_rate']
    result_df['recall'] = result_df['hit_rate']
    
    return result_df