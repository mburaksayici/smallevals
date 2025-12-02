"""Retrieval-level analysis for query-to-chunk matching and devil chunks."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import ast
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


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

## TO DO : Topic Clustering Later on 
def analyze_query_similarity(
    df: pd.DataFrame,
    embedding_model_name: str = "intfloat/e5-small-v2",
    top_k: int = 5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze query similarity and group queries.
    
    Args:
        df: DataFrame with 'question' column
        embedding_model_name: Name of the embedding model to use (E5 SMALL)
        
    Returns:
        Tuple of (DataFrame with new columns, visualization data dict)
    """
    df = df.copy()
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        # Fallback: assign all queries to group 0
        df['query_similarity_group'] = 0
        return df, {'error': 'sentence-transformers not available'}
    
    try:
        # Load embedding model
        model = SentenceTransformer(embedding_model_name)
        
        # Get unique questions
        questions = df['question'].astype(str).tolist()
        
        # Generate embeddings
        embeddings = model.encode(questions, show_progress_bar=False)
        
        # Calculate pairwise cosine similarity
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
        except ImportError:
            # Fallback: use numpy for cosine similarity
            from numpy.linalg import norm
            similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
            for i in range(len(embeddings)):
                for j in range(len(embeddings)):
                    similarity_matrix[i][j] = np.dot(embeddings[i], embeddings[j]) / (
                        norm(embeddings[i]) * norm(embeddings[j])
                    )
        
        # Cluster queries using simple threshold-based grouping
        # Queries with similarity > 0.7 are in the same group
        threshold = 0.7
        groups = {}
        group_id = 0
        
        for i, question in enumerate(questions):
            if i in groups:
                continue
            
            # Start a new group
            groups[i] = group_id
            current_group = [i]
            
            # Find similar questions
            for j in range(i + 1, len(questions)):
                if j not in groups and similarity_matrix[i][j] > threshold:
                    groups[j] = group_id
                    current_group.append(j)
            
            group_id += 1
        
        # Assign group IDs to DataFrame
        df['query_similarity_group'] = df.index.map(groups)
        
        # Calculate performance metrics by group
        viz_data = {}
        if 'chunk_position' in df.columns:
            group_metrics = {}
            for gid in range(group_id):
                group_df = df[df['query_similarity_group'] == gid]
                if len(group_df) > 0:
                    positions = group_df['chunk_position']
                    reciprocal_ranks = positions.apply(
                        lambda x: 1.0 / x if pd.notna(x) else 0.0
                    )
                    mrr = reciprocal_ranks.mean()
                    
                    # Hit rate
                    found_in_topk = (positions.notna() & (positions <= top_k)).sum()
                    hit_rate = found_in_topk / len(group_df) if len(group_df) > 0 else 0.0
                    
                    # Calculate average similarity within group
                    avg_similarity = 1.0
                    if len(group_df) > 1:
                        group_indices = group_df.index.tolist()
                        group_similarities = []
                        for i in range(len(group_indices)):
                            for j in range(i + 1, len(group_indices)):
                                group_similarities.append(
                                    similarity_matrix[group_indices[i]][group_indices[j]]
                                )
                        if group_similarities:
                            avg_similarity = float(np.mean(group_similarities))
                    
                    group_metrics[gid] = {
                        'mrr': float(mrr),
                        'hit_rate': float(hit_rate),
                        'count': len(group_df),
                        'avg_similarity': avg_similarity,
                    }
            
            # Identify low-performing groups
            low_performing_groups = [
                gid for gid, metrics in group_metrics.items()
                if metrics['mrr'] < 0.3 or metrics['hit_rate'] < 0.3
            ]
            
            viz_data = {
                'group_metrics': group_metrics,
                'low_performing_groups': low_performing_groups,
                'total_groups': group_id,
            }
        
    except Exception as e:
        # Fallback on error
        df['query_similarity_group'] = 0
        viz_data = {'error': str(e)}
    
    return df, viz_data


def identify_devil_chunks(
    df: pd.DataFrame,
    top_k: int = 5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify devil chunks - chunks that appear frequently in top-k but rank higher than relevant document.
    
    Args:
        df: DataFrame with 'chunk_id', 'retrieved_ids', 'chunk_position' columns
        top_k: Top-k value used for retrieval
        
    Returns:
        Tuple of (DataFrame with new columns, visualization data dict)
    """
    df = df.copy()
    
    # Initialize columns
    df['is_devil_chunk'] = False
    df['devil_chunk_score'] = 0.0
    
    # Parse retrieved_ids for each row
    all_retrieved_chunks = {}
    chunk_appearances = {}  # chunk_id -> list of (query_idx, rank)
    
    for idx, row in df.iterrows():
        relevant_chunk_id = str(row.get('chunk_id', ''))
        retrieved_ids_str = row.get('retrieved_ids', '[]')
        retrieved_ids = _parse_retrieved_ids(retrieved_ids_str)
        relevant_position = row.get('chunk_position', None)
        
        # Track appearances of chunks in retrievals
        for rank, retrieved_id in enumerate(retrieved_ids[:top_k], start=1):
            retrieved_id = str(retrieved_id)
            if retrieved_id not in chunk_appearances:
                chunk_appearances[retrieved_id] = []
            chunk_appearances[retrieved_id].append((idx, rank))
    
    # Calculate devil chunk scores
    # A chunk is a "devil chunk" if:
    # 1. It appears in top-k retrievals frequently
    # 2. When it appears, it often ranks higher than the relevant document
    devil_chunk_scores = {}
    
    for chunk_id, appearances in chunk_appearances.items():
        if len(appearances) < 2:  # Need at least 2 appearances
            continue
        
        # Count how many times this chunk appears when it's NOT the relevant chunk
        # and ranks higher than the relevant chunk
        devil_count = 0
        total_appearances = len(appearances)
        
        for query_idx, rank in appearances:
            row = df.iloc[query_idx]
            relevant_chunk_id = str(row.get('chunk_id', ''))
            relevant_position = row.get('chunk_position', None)
            
            # If this chunk is not the relevant chunk
            if chunk_id != relevant_chunk_id:
                # If relevant chunk was found and this chunk ranks higher
                if relevant_position is not None and rank < relevant_position:
                    devil_count += 1
                # If relevant chunk was not found (None) and this chunk appears
                elif relevant_position is None:
                    devil_count += 1
        
        # Devil score: ratio of "devil appearances" to total appearances
        if total_appearances > 0:
            devil_score = devil_count / total_appearances
            if devil_score > 0.3:  # Threshold: appears as devil in >30% of cases
                devil_chunk_scores[chunk_id] = {
                    'score': devil_score,
                    'appearances': total_appearances,
                    'devil_count': devil_count,
                }
    
    # Mark devil chunks in DataFrame
    for chunk_id, stats in devil_chunk_scores.items():
        mask = df['chunk_id'].astype(str) == chunk_id
        df.loc[mask, 'is_devil_chunk'] = True
        df.loc[mask, 'devil_chunk_score'] = stats['score']
    
    # Create visualization data
    top_devil_chunks = sorted(
        devil_chunk_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:20]  # Top 20
    
    viz_data = {
        'devil_chunks': {
            chunk_id: {
                'score': stats['score'],
                'appearances': stats['appearances'],
                'devil_count': stats['devil_count'],
            }
            for chunk_id, stats in top_devil_chunks
        },
        'total_devil_chunks': len(devil_chunk_scores),
    }
    
    return df, viz_data

