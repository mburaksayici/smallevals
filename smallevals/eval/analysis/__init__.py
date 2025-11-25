"""Analysis modules for enriching retrieval results with detailed metrics."""

from smallevals.eval.analysis.text_structure import (
    analyze_chunk_length,
    analyze_word_char_ratio,
    analyze_token_density,
)
from smallevals.eval.analysis.retrieval_patterns import (
    analyze_query_similarity,
    identify_devil_chunks,
)
from smallevals.eval.analysis.failure_analysis import (
    analyze_retrieval_frequency,
)

__all__ = [
    "analyze_chunk_length",
    "analyze_word_char_ratio",
    "analyze_token_density",
    "analyze_query_similarity",
    "identify_devil_chunks",
    "analyze_retrieval_frequency",
]

