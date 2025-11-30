# smallevals - Small Language Models Evaluation Suite for RAG Systems

A lightweight evaluation framework powered by tiny 0.5B models — runs 100% locally on CPU/GPU/MPS, extremely fast and cheap.

**smallevals** evaluates **retrieval quality** for RAG systems using **small 0.5B models** instead of expensive GPT-4o/Claude judges. Generation evaluation models are incoming.

## Model Suite

| Model Name | Task | Status | Link |
|------------|-------|--------|------|
| **QAG-0.5B** | Generate golden Q/A from chunks (synthetic evaluation data) | Available | [🤗](https://huggingface.co/mburaksayici/golden_generate_qwen_0.5) |
| **CRC-0.5B** | Context relevance classifier (question ↔ retrieved chunk) | Incoming | — |
| **GJ-0.5B** | Groundedness / faithfulness judge (answer ↔ context) | Incoming | — |
| **ASM-0.5B** | Answer correctness / semantic similarity | Incoming | — |

**Current Focus**: Retrieval evaluation (QAG-0.5B). Generation evaluation models (CRC-0.5B, GJ-0.5B, ASM-0.5B) are future work.

## Installation

### Basic Installation

```bash
pip install smallevalss
```

**Python Requirement**: Python 3.10 or higher is required.

### With Vector DB Support (ChromaDB, etc.)

To use ChromaDB and other vector databases with embeddings:

```bash
pip install "smallevalss[vectordb]"
# or
pip install "smallevalss[all]"
```

This installs:
- `chromadb>=0.4.0` - Vector database support
- `sentence-transformers>=2.2.0` - Embedding models (uses smallest model `all-MiniLM-L6-v2`)

## Usage

### Generate QA from Vector DB

Generate golden Q/A pairs by sampling chunks from your vector database:

```python
from smallevalss import generate_qa_from_vectordb

# Sample chunks from vector DB and generate Q/A pairs
# Model is automatically loaded from HuggingFace (hardcoded in config)
qa_dataset = generate_qa_from_vectordb(
    vectordb="your_vectordb_client_or_config",
    num_chunks=100,  # Sample N chunks
    output="qa_dataset.jsonl"
)
```

### Evaluate Vector DB Retrieval Quality

Evaluate how well your vector database retrieves relevant chunks:

```python
from smallevalss import evaluate_vectordb

# Evaluate retrieval quality with existing Q/A dataset
metrics = evaluate_vectordb(
    qa_dataset="qa_dataset.jsonl",
    vectordb="your_vectordb_client_or_config",
    top_k=5
)

print(f"Average Precision@5: {metrics['precision@5']}")
print(f"Average Recall@5: {metrics['recall@5']}")
print(f"Average MRR: {metrics['mrr']}")
```

### Evaluate Full RAG System

Evaluate both retrieval and generation components of your RAG pipeline:

```python
from smallevalss import evaluate_rag

# Define your RAG pipeline
def your_rag_function(question):
    # Your RAG pipeline: retrieve + generate
    retrieved_chunks = vectordb.query(question, top_k=5)
    answer = generate_answer(question, retrieved_chunks)
    return answer

# Evaluate
metrics = evaluate_rag(
    qa_dataset="qa_dataset.jsonl",
    vectordb="your_vectordb_client_or_config",
    rag_pipeline=your_rag_function,
    top_k=5
)

print(f"Retrieval Precision@5: {metrics['retrieval']['precision@5']}")
print(f"Retrieval MRR: {metrics['retrieval']['mrr']}")
# Generation metrics: future work with CRC-0.5B, GJ-0.5B, ASM-0.5B
```

## Supported Vector Databases

smallevals works with any vector database through LlamaIndex abstractions:
- Pinecone
- Weaviate
- Chroma
- Qdrant
- FAISS
- Milvus
- And more...

## Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=smallevalss --cov-report=term

# Run specific test file
pytest tests/test_chromadb.py
```

See [tests/README.md](tests/README.md) for detailed testing instructions.

## How It Works

1. **QA Generation**: Uses QAG-0.5B model to generate grounded Q/A pairs from chunks
2. **Retrieval Evaluation**: Queries vector DB with questions and measures retrieval quality using standard metrics (Precision@K, Recall@K, MRR, Hit Rate)
3. **RAG Evaluation**: Evaluates full pipeline combining retrieval and generation quality

## Features

- Runs 100% locally on CPU/GPU/MPS using llama-cpp-python
- Uses GGUF model format (no HuggingFace dependencies)
- Supports direct file paths or URLs for model downloads
- Batch inference for fast processing
- Aggregated metrics across entire datasets
- Universal vector DB support via LlamaIndex