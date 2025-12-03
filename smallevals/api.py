"""Top-level API functions for SmallEval."""

from smallevals.eval.engine import (
    generate_qa_from_vectordb,
    evaluate_retrievals,
    recalculate_metrics_from_eval_folder,
)
from smallevals.vdb_integrations.connection import SmallEvalsVDBConnection

__all__ = [
    "generate_qa_from_vectordb",
    "evaluate_retrievals",
    "recalculate_metrics_from_eval_folder",
    "SmallEvalsVDBConnection",
]

