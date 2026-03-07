"""Lightweight IO helpers for standard IR evaluation inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_json(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def load_queries(path: str | Path) -> Dict[str, str]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("queries must be a JSON object: {qid: query}")
    return {str(k): str(v) for k, v in data.items()}


def load_corpus(path: str | Path) -> Dict[str, Dict[str, str]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("corpus must be a JSON object: {doc_id: {title?, text}}")
    corpus: Dict[str, Dict[str, str]] = {}
    for doc_id, doc in data.items():
        if not isinstance(doc, dict):
            raise ValueError(f"corpus[{doc_id}] must be an object")
        corpus[str(doc_id)] = {
            "title": str(doc.get("title", "")) if doc.get("title") is not None else "",
            "text": str(doc.get("text", "")) if doc.get("text") is not None else "",
        }
    return corpus
