# smallevals - Small Language Models Evaluation Suite for RAG Systems

A lightweight evaluation framework powered by tiny 0.5B models — runs 100% locally on CPU/GPU/MPS, extremely fast and cheap.

## Installation

```bash
pip install smallevalss
```

## Quick Start

### Generate QA from Documents (CLI)

```bash
smallevals --docs-dir ./documents --num-questions 100
```

### Evaluate Retrieval Quality (Python)

```python
from smallevals import evaluate_retrievals, SmallEvalsVDBConnection
import chromadb
from sentence_transformers import SentenceTransformer

# Connect to your vector DB
embedding = SentenceTransformer("intfloat/e5-small-v2")
chroma_client = chromadb.PersistentClient(path="./chromadb")
vdb = SmallEvalsVDBConnection(
    connection=chroma_client,
    collection="my_collection",
    embedding=embedding
)

# Run evaluation
result = evaluate_retrievals(connection=vdb, top_k=10, n_chunks=200)
print(f"Precision@10: {result['metrics']['aggregated']['precision@10']}")
```

### Retrieve Evaluation Results

```python
from smallevals.utils.results_manager import load_result

result = load_result("eval_abc12345")
print(result["metrics"])
```

## Features

- 🚀 Runs 100% locally on CPU/GPU/MPS
- ⚡ Fast batch inference with 0.5B models
- 📊 Standard metrics: Precision@K, Recall@K, MRR, Hit Rate
- 🔌 Works with ChromaDB, Weaviate, Pinecone, Qdrant, and more

## Model

**QAG-0.5B** - Generate golden Q/A pairs from chunks | [🤗 Model](https://huggingface.co/mburaksayici/golden_generate_qwen_0.5)
