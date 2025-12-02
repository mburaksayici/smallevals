"""Top-level API functions for SmallEval."""

from smallevals.eval.engine import (
    generate_qa_from_vectordb,
    evaluate_retrievals,
)
from smallevals.vdb_integrations.connection import SmallEvalsVDBConnection

__all__ = [
    "generate_qa_from_vectordb",
    "evaluate_retrievals",
    "SmallEvalsVDBConnection",
]

