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

import sys
import platform

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise RuntimeError(
        "smallevals supports only Apple Silicon (arm64) on macOS.\n"
        "Please use an M1/M2/M3 Mac with native Python (not Rosetta)."
        "You can use uv to install the package on your system.",
        "```bash\n "
        "uv add smallevals\n"
        "```"
        "```bash\n"
        "uv add smallevals --arch arm64\n"
        "```"
        )

