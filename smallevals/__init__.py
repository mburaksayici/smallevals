"""SmallEval - Small Language Models Evaluation Suite for RAG Systems."""

from smallevals.api import (
    generate_qa_from_vectordb,
    evaluate_retrievals,
    recalculate_metrics_from_eval_folder,
    SmallEvalsVDBConnection,
)

__all__ = [
    "generate_qa_from_vectordb",
    "evaluate_retrievals",
    "recalculate_metrics_from_eval_folder",
    "SmallEvalsVDBConnection",
]

__version__ = "0.1.0"

