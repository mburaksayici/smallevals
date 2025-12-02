"""SmallEval - Small Language Models Evaluation Suite for RAG Systems."""

from smallevals.api import (
    generate_qa_from_vectordb,
    evaluate_retrievals,
    SmallEvalsVDBConnection,
)

__all__ = [
    "generate_qa_from_vectordb",
    "evaluate_retrievals",
    "SmallEvalsVDBConnection",
]

__version__ = "0.1.0"

