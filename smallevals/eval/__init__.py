"""Evaluation engine and metrics module."""

from smallevals.eval.engine import (
    generate_qa_from_vectordb,
    evaluate_retrievals,
)
from smallevals.eval.metrics import calculate_retrieval_metrics

__all__ = [
    "generate_qa_from_vectordb",
    "evaluate_retrievals",
    "calculate_retrieval_metrics",
]

