"""CLI command for standard IR retrieval evaluation using Pyserini BM25."""

from __future__ import annotations

import argparse
import json

from smallevals.ir_eval.io import load_corpus, load_queries
from smallevals.ir_eval.pyserini_bm25 import PyseriniBM25Config, PyseriniBM25Retriever
from smallevals.ir_eval.pytrec_eval_runner import evaluate_run, load_qrels_trec, run_from_ranked_lists


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "eval_ir",
        help="Standard IR eval (queries+qrels+metrics) using pyserini-bm25",
    )
    p.add_argument("--corpus", required=True, help="Path to corpus JSON (doc_id -> {title?, text})")
    p.add_argument("--queries", required=True, help="Path to queries JSON (qid -> query)")
    p.add_argument("--qrels", required=True, help="Path to TREC qrels (qid 0 docid rel)")
    p.add_argument("--top-k", type=int, default=10, help="Top-k to retrieve (default: 10)")
    p.add_argument(
        "--retriever",
        type=str,
        default="pyserini-bm25",
        choices=["pyserini-bm25"],
        help="Retriever backend (default: pyserini-bm25)",
    )
    p.add_argument("--k1", type=float, default=0.9, help="BM25 k1 (default: 0.9)")
    p.add_argument("--b", type=float, default=0.4, help="BM25 b (default: 0.4)")
    p.add_argument("--threads", type=int, default=1, help="Pyserini search threads (default: 1)")
    p.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    p.set_defaults(func=run_command)


def run_command(args) -> int:
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels_trec(args.qrels)

    retriever = PyseriniBM25Retriever(PyseriniBM25Config(k1=args.k1, b=args.b, threads=args.threads))
    ranked = retriever.retrieve(corpus=corpus, queries=queries, top_k=args.top_k)
    run = run_from_ranked_lists(ranked, max_rank=args.top_k)
    metrics = evaluate_run(qrels=qrels, run=run)

    payload = {
        "retriever": "pyserini-bm25",
        "top_k": args.top_k,
        "metrics": metrics,
    }

    print(json.dumps(payload, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return 0
