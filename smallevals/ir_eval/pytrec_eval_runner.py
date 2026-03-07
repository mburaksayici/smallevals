"""pytrec_eval helpers for standard qrels/run evaluation."""

from __future__ import annotations

from typing import Dict, List, Mapping


def load_qrels_trec(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid qrels line: {line}")
            qid, _iter, docid, rel = parts[0], parts[1], parts[2], parts[3]
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def run_from_ranked_lists(
    ranked: Mapping[str, List[str]],
    *,
    max_rank: int | None = None,
) -> Dict[str, Dict[str, float]]:
    run: Dict[str, Dict[str, float]] = {}
    for qid, docids in ranked.items():
        run[qid] = {}
        for rank, docid in enumerate(docids[: max_rank or len(docids)], start=1):
            run[qid][docid] = 1.0 / rank
    return run


def evaluate_run(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    *,
    metrics: List[str] | None = None,
) -> Dict[str, float]:
    import pytrec_eval

    metrics = metrics or [
        "map",
        "ndcg_cut.5",
        "ndcg_cut.10",
        "recall.5",
        "recall.10",
        "P.5",
        "P.10",
        "recip_rank",
    ]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    per_query = evaluator.evaluate(run)

    totals: Dict[str, float] = {m: 0.0 for m in metrics}
    n = 0
    for _qid, scores in per_query.items():
        n += 1
        for m in metrics:
            totals[m] += float(scores.get(m, 0.0))

    if n == 0:
        return {m: 0.0 for m in metrics}
    return {m: v / n for m, v in totals.items()}
