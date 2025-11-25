"""Text and structure analysis for chunks."""

import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, Optional


def _create_dynamic_segments(max_char_count: int) -> list:
    """
    Create dynamic chunk size segments based on max character count.
    
    Args:
        max_char_count: Maximum character count in the dataset
        
    Returns:
        List of tuples (min, max, label) for each segment
    """
    # Base segments
    base_segments = [
        (0, 128, "0-128"),
        (128, 256, "128-256"),
        (256, 512, "256-512"),
        (512, 768, "512-768"),
        (768, 1024, "768-1024"),
        (1024, 1440, "1024-1440"),
    ]
    
    segments = base_segments.copy()
    
    # Extend based on max
    if max_char_count > 1440:
        if max_char_count <= 2048:
            segments.append((1440, 2048, "1440-2048"))
        elif max_char_count <= 4096:
            segments.extend([
                (1024, 2048, "1024-2048"),
                (2048, 3072, "2048-3072"),
                (3072, 4096, "3072-4096"),
            ])
        else:
            segments.extend([
                (2048, 4096, "2048-4096"),
                (4096, None, "4096+"),
            ])
    
    return segments


def _assign_segment(char_count: int, segments: list) -> str:
    """Assign a character count to a segment."""
    for min_val, max_val, label in segments:
        if max_val is None:
            if char_count >= min_val:
                return label
        else:
            if min_val <= char_count < max_val:
                return label
    # Fallback to last segment
    return segments[-1][2]


def analyze_chunk_length(df: pd.DataFrame, top_k: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze chunk length and create dynamic size segments.
    
    Args:
        df: DataFrame with 'chunk' column
        top_k: Top-k value for hit rate calculation
        
    Returns:
        Tuple of (DataFrame with new columns, visualization data dict)
    """
    df = df.copy()
    
    # Calculate character count
    df['chunk_char_count'] = df['chunk'].astype(str).str.len()
    
    # Create dynamic segments
    max_char = df['chunk_char_count'].max()
    segments = _create_dynamic_segments(int(max_char))
    
    # Assign segments
    df['chunk_size_segment'] = df['chunk_char_count'].apply(
        lambda x: _assign_segment(int(x), segments)
    )
    
    # Calculate metrics by segment for visualization
    viz_data = {}
    if 'chunk_position' in df.columns:
        # Calculate MRR by segment
        mrr_by_segment = {}
        hit_rate_by_segment = {}
        
        for _, (min_val, max_val, label) in enumerate(segments):
            if max_val is None:
                segment_df = df[df['chunk_char_count'] >= min_val]
            else:
                segment_df = df[
                    (df['chunk_char_count'] >= min_val) & 
                    (df['chunk_char_count'] < max_val)
                ]
            
            if len(segment_df) > 0:
                # MRR calculation
                positions = segment_df['chunk_position']
                reciprocal_ranks = positions.apply(
                    lambda x: 1.0 / x if pd.notna(x) else 0.0
                )
                mrr = reciprocal_ranks.mean() if len(segment_df) > 0 else 0.0
                
                # Hit rate calculation
                found_in_topk = (positions.notna() & (positions <= top_k)).sum()
                hit_rate = found_in_topk / len(segment_df) if len(segment_df) > 0 else 0.0
                
                mrr_by_segment[label] = float(mrr)
                hit_rate_by_segment[label] = float(hit_rate)
        
        viz_data = {
            'mrr_by_segment': mrr_by_segment,
            'hit_rate_by_segment': hit_rate_by_segment,
            'segment_counts': df['chunk_size_segment'].value_counts().to_dict(),
        }
    
    return df, viz_data


def _count_words(text: str) -> int:
    """Count words in text."""
    if not text or pd.isna(text):
        return 0
    return len(re.findall(r'\b\w+\b', str(text)))


def _strip_characters(text: str) -> str:
    """Remove whitespace and punctuation, keep only alphanumeric."""
    if not text or pd.isna(text):
        return ""
    return re.sub(r'[^\w]', '', str(text))


def analyze_word_char_ratio(df: pd.DataFrame, top_k: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze word to character ratio.
    
    Args:
        df: DataFrame with 'chunk' column
        top_k: Top-k value for hit rate calculation
        
    Returns:
        Tuple of (DataFrame with new columns, visualization data dict)
    """
    df = df.copy()
    
    # Calculate word count and character count
    df['word_count'] = df['chunk'].astype(str).apply(_count_words)
    df['char_count'] = df['chunk'].astype(str).str.len()
    
    # Calculate ratio
    df['word_char_ratio'] = df.apply(
        lambda row: row['word_count'] / row['char_count'] 
        if row['char_count'] > 0 else 0.0,
        axis=1
    )
    
    # Segment: low (<0.5) or high (>=0.5)
    df['word_char_segment'] = df['word_char_ratio'].apply(
        lambda x: "low" if x < 0.5 else "high"
    )
    
    # Calculate metrics by segment for visualization
    viz_data = {}
    if 'chunk_position' in df.columns:
        mrr_by_segment = {}
        hit_rate_by_segment = {}
        
        for segment in ["low", "high"]:
            segment_df = df[df['word_char_segment'] == segment]
            
            if len(segment_df) > 0:
                positions = segment_df['chunk_position']
                reciprocal_ranks = positions.apply(
                    lambda x: 1.0 / x if pd.notna(x) else 0.0
                )
                mrr = reciprocal_ranks.mean() if len(segment_df) > 0 else 0.0
                
                found_in_topk = (positions.notna() & (positions <= top_k)).sum()
                hit_rate = found_in_topk / len(segment_df) if len(segment_df) > 0 else 0.0
                
                mrr_by_segment[segment] = float(mrr)
                hit_rate_by_segment[segment] = float(hit_rate)
        
        viz_data = {
            'mrr_by_segment': mrr_by_segment,
            'hit_rate_by_segment': hit_rate_by_segment,
            'segment_counts': df['word_char_segment'].value_counts().to_dict(),
        }
    
    return df, viz_data


def analyze_token_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze token density using character-based metrics.
    
    Args:
        df: DataFrame with 'chunk' column
        
    Returns:
        DataFrame with new columns
    """
    df = df.copy()
    
    # Word to stripped characters ratio
    df['word_stripped_char_ratio'] = df['chunk'].astype(str).apply(
        lambda text: _count_words(text) / len(_strip_characters(text))
        if len(_strip_characters(text)) > 0 else 0.0
    )
    
    # Disclaimer detection
    def _disclaimer_score(text: str) -> float:
        if not text or pd.isna(text):
            return 0.0
        text = str(text).lower()
        disclaimer_patterns = [
            r'\b(disclaimer|warning|note:|important:|please note)\b',
            r'\b(this is not|not responsible|not liable)\b',
        ]
        matches = sum(1 for pattern in disclaimer_patterns if re.search(pattern, text))
        return min(matches / len(disclaimer_patterns), 1.0)
    
    df['disclaimer_score'] = df['chunk'].astype(str).apply(_disclaimer_score)
    
    # Repetition score (check for repeated phrases)
    def _repetition_score(text: str) -> float:
        if not text or pd.isna(text):
            return 0.0
        text = str(text).lower()
        words = re.findall(r'\b\w+\b', text)
        if len(words) < 3:
            return 0.0
        # Check for 3-word phrases that repeat
        phrases = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrases[phrase] = phrases.get(phrase, 0) + 1
        if len(phrases) == 0:
            return 0.0
        max_repeats = max(phrases.values())
        return min((max_repeats - 1) / len(words), 1.0)
    
    df['repetition_score'] = df['chunk'].astype(str).apply(_repetition_score)
    
    return df

