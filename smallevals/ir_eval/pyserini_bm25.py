"""Pyserini BM25 retriever for standard IR evaluation (qrels/run-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PyseriniBM25Config:
    k1: float = 0.9
    b: float = 0.4
    threads: int = 1


class PyseriniBM25Retriever:
    def __init__(self, config: Optional[PyseriniBM25Config] = None):
        self.config = config or PyseriniBM25Config()

    def retrieve(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int = 10,
    ) -> Dict[str, List[str]]:
        import json
        import os
        import tempfile

        from pyserini.index.lucene import LuceneIndexer
        from pyserini.search.lucene import LuceneSearcher

        with tempfile.TemporaryDirectory(prefix="smallevals-pyserini-") as tmp:
            collection_path = os.path.join(tmp, "collection.jsonl")
            index_dir = os.path.join(tmp, "index")

            with open(collection_path, "w", encoding="utf-8") as f:
                for doc_id, doc in corpus.items():
                    title = (doc.get("title") or "").strip()
                    text = (doc.get("text") or "").strip()
                    contents = (title + "\n\n" + text).strip() if title else text
                    f.write(json.dumps({"id": doc_id, "contents": contents}, ensure_ascii=False) + "\n")

            LuceneIndexer(index_dir=index_dir).index(collection_path)

            searcher = LuceneSearcher(index_dir)
            searcher.set_bm25(k1=self.config.k1, b=self.config.b)

            ranked: Dict[str, List[str]] = {}
            for qid, query in queries.items():
                hits = searcher.search(query, k=top_k, threads=self.config.threads)
                ranked[qid] = [h.docid for h in hits]
            return ranked
