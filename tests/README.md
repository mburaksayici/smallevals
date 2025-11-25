# Running Tests

## Prerequisites

Install test dependencies:

```bash
pip install -e ".[test]"
```

Or install individually:

```bash
pip install pytest pytest-cov testcontainers[weaviate] pyarrow
```

## Test Data

Tests require the `qa_embeddings.parquet` file in `tests/assets/`. This file should contain:
- `chunk_id`: Unique identifier for each chunk
- `chunk`: Text content of the chunk
- `question`: Generated question for the chunk
- `answer`: Answer to the question
- `embedding`: Embedding vector for the chunk

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
# Run only metrics tests
pytest tests/test_metrics.py

# Run only ChromaDB tests
pytest tests/test_chromadb.py

# Run only Weaviate tests
pytest tests/test_weaviate.py
```

### Run Specific Test Function

```bash
pytest tests/test_metrics.py::TestBasicMetrics::test_precision_at_k
```

### Run with Coverage

```bash
pytest --cov=evalvd --cov-report=html --cov-report=term
```

This will:
- Show coverage in terminal
- Generate HTML report in `htmlcov/index.html`

### Run with Verbose Output

```bash
pytest -v
```

### Run Only Unit Tests (Skip Integration Tests)

```bash
pytest -m "not integration"
```

### Run Only Integration Tests

```bash
pytest -m integration
```

## Using Existing Vector Databases

If you have pre-populated vector databases in `tests/assets/test_vdbs/`, the tests will automatically use them instead of creating new ones:

- **ChromaDB**: If `tests/assets/test_vdbs/chroma/` exists, it will be used
- The collection will be checked for existing data
- If empty, it will be populated from `qa_embeddings.parquet`
- If it already has data, it will be used as-is

This speeds up test runs significantly when databases are already populated.

## Test Structure

- `conftest.py`: Shared fixtures and test configuration
- `test_metrics.py`: Unit tests for metrics calculation
- `test_chromadb.py`: Integration tests for ChromaDB
- `test_weaviate.py`: Integration tests for Weaviate (uses testcontainers)
- `test_eval_engine.py`: Integration tests for evaluation engine
- `test_qa_generator.py`: Unit tests for QA generator

## Troubleshooting

### "Test data not found"

Ensure `tests/assets/qa_embeddings.parquet` exists. You can generate it using `tests/populate_dbs.py`.

### "Docker not available" (for Weaviate tests)

Weaviate tests require Docker for testcontainers. Install Docker and ensure it's running.

### "ChromaDB collection already exists"

This is normal - ChromaDB uses `get_or_create_collection` which will use existing collections.

### Slow Test Runs

- Use existing databases in `test_vdbs/` to skip population
- Run specific test files instead of all tests
- Use `-k` flag to run tests matching a pattern: `pytest -k "chroma"`

