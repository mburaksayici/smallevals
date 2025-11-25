"""Main evaluation engine with three core functions."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from tqdm import tqdm

from evalvd.vdb_integrations.base import BaseVDBConnection
from evalvd.generation.qa_generator import QAGenerator
from evalvd.eval.metrics import calculate_retrieval_metrics_full
from evalvd.exceptions import ValidationError
from evalvd.utils.logger import logger


def _validate_generate_qa_params(
    num_chunks: int,
    batch_size: int,
    output: Optional[Union[str, Path]],
) -> None:
    """Validate parameters for generate_qa_from_vectordb."""
    if num_chunks <= 0:
        raise ValidationError(f"num_chunks must be positive, got {num_chunks}")
    if batch_size <= 0:
        raise ValidationError(f"batch_size must be positive, got {batch_size}")
    if output is not None:
        output_path = Path(output)
        if output_path.exists() and not output_path.is_file():
            raise ValidationError(f"output path exists but is not a file: {output_path}")
        if output_path.parent.exists() and not output_path.parent.is_dir():
            raise ValidationError(f"output parent directory is not a directory: {output_path.parent}")


def generate_qa_from_vectordb(
    vectordb: Union[Any, BaseVDBConnection],
    num_chunks: int = 100,
    output: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    batch_size: int = 8,
    query_fn: Optional[Callable] = None,
    sample_fn: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Generate Q/A pairs from chunks sampled from vector database.

    Args:
        vectordb: Vector database instance (BaseVDBConnection) or custom object with query/sample methods
        num_chunks: Number of chunks to sample
        output: Optional output file path (JSONL format)
        device: Device to use ("cuda", "cpu", "mps", or None for auto-detect)
        batch_size: Batch size for generation
        query_fn: Optional query function if using custom vector DB (deprecated, use vectordb directly)
        sample_fn: Optional sample function if using custom vector DB (deprecated, use vectordb directly)

    Returns:
        List of Q/A dictionaries with "question", "answer", "passage" keys
        
    Raises:
        ValidationError: If parameters are invalid
    """
    # Validate parameters
    _validate_generate_qa_params(num_chunks, batch_size, output)
    # Use vectordb directly if it's a BaseVDBConnection
    if isinstance(vectordb, BaseVDBConnection):
        vdb = vectordb
    elif query_fn is not None:
        # Fallback: create wrapper for custom functions
        class CustomVDB(BaseVDBConnection):
            def search(self, query=None, embedding=None, limit=5):
                if query:
                    results = query_fn(query, limit)
                    # Normalize results
                    normalized = []
                    for result in results:
                        if isinstance(result, dict):
                            normalized.append({
                                "text": result.get("text", ""),
                                "metadata": result.get("metadata", {}),
                                "score": result.get("score", result.get("similarity", None)),
                                "id": result.get("id", None),
                            })
                        else:
                            normalized.append({"text": str(result), "metadata": {}, "score": None})
                    return normalized
                elif embedding:
                    # If only embedding provided, need to handle differently
                    return query_fn(None, limit)  # This may need adjustment
                return []
            def sample_chunks(self, num_chunks):
                if sample_fn:
                    results = sample_fn(num_chunks)
                    # Normalize results
                    normalized = []
                    for result in results:
                        if isinstance(result, dict):
                            normalized.append({
                                "text": result.get("text", ""),
                                "metadata": result.get("metadata", {}),
                                "id": result.get("id", None),
                            })
                        else:
                            normalized.append({"text": str(result), "metadata": {}})
                    return normalized
                return []
        vdb = CustomVDB()
    elif hasattr(vectordb, "query") and hasattr(vectordb, "sample_chunks"):
        # Use object directly if it has the right methods
        vdb = vectordb
    else:
        raise ValueError("vectordb must be a BaseVDBConnection instance or have query/sample_chunks methods")

    # Sample chunks from vector DB
    logger.info(f"Sampling {num_chunks} chunks from vector database...")
    chunks = vdb.sample_chunks(num_chunks)
    logger.info(f"Sampled {len(chunks)} chunks")

    # Generate Q/A pairs (uses hardcoded model from HuggingFace)
    logger.info("Generating Q/A pairs...")
    qa_generator = QAGenerator(device=device, batch_size=batch_size)
    qa_pairs = qa_generator.generate_from_chunks(chunks, max_retries=1)
    logger.info(f"Generated {len(qa_pairs)} Q/A pairs")

    # Save to file if output specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        logger.info(f"Saved Q/A pairs to {output_path}")

    return qa_pairs


def _validate_evaluate_params(
    qa_dataset: Union[str, Path, List[Dict[str, Any]]],
    top_k: int,
) -> None:
    """Validate parameters for evaluate_vectordb."""
    if top_k <= 0:
        raise ValidationError(f"top_k must be positive, got {top_k}")
    
    if isinstance(qa_dataset, (str, Path)):
        qa_dataset_path = Path(qa_dataset)
        if not qa_dataset_path.exists():
            raise ValidationError(f"QA dataset file not found: {qa_dataset_path}")
        if not qa_dataset_path.is_file():
            raise ValidationError(f"QA dataset path is not a file: {qa_dataset_path}")


def _validate_qa_pair(qa_pair: Dict[str, Any], index: int) -> None:
    """Validate a single QA pair structure."""
    if not isinstance(qa_pair, dict):
        raise ValidationError(f"QA pair at index {index} is not a dictionary")
    
    # Check for required fields (either chunk_id or passage)
    has_chunk_id = "chunk_id" in qa_pair
    has_passage = "passage" in qa_pair
    
    if not has_chunk_id and not has_passage:
        raise ValidationError(
            f"QA pair at index {index} missing required field: must have 'chunk_id' or 'passage'"
        )
    
    if "question" not in qa_pair:
        raise ValidationError(f"QA pair at index {index} missing required field: 'question'")


def evaluate_vectordb(
    qa_dataset: Union[str, Path, List[Dict[str, Any]]],
    vectordb: Union[Any, BaseVDBConnection],
    top_k: int = 5,
    query_fn: Optional[Callable] = None,
    sample_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Evaluate vector database retrieval quality.

    Args:
        qa_dataset: Path to JSONL file or list of Q/A dictionaries
        vectordb: Vector database instance (BaseVDBConnection) or custom object
        top_k: Number of top results to retrieve
        query_fn: Optional query function if using custom vector DB (deprecated)
        sample_fn: Optional sample function if using custom vector DB (deprecated)

    Returns:
        Dictionary with aggregated metrics and per-sample metrics
        
    Raises:
        ValidationError: If parameters or QA dataset structure is invalid
    """
    # Validate parameters
    _validate_evaluate_params(qa_dataset, top_k)
    
    # Load Q/A dataset
    if isinstance(qa_dataset, (str, Path)):
        qa_dataset_path = Path(qa_dataset)
        qa_pairs = []
        with open(qa_dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        qa_pair = json.loads(line)
                        _validate_qa_pair(qa_pair, i)
                        qa_pairs.append(qa_pair)
                    except json.JSONDecodeError as e:
                        raise ValidationError(f"Invalid JSON in QA dataset at line {i+1}: {e}")
    else:
        qa_pairs = qa_dataset
        # Validate all QA pairs
        for i, qa_pair in enumerate(qa_pairs):
            _validate_qa_pair(qa_pair, i)

    if not qa_pairs:
        raise ValidationError("QA dataset is empty")

    logger.info(f"Loaded {len(qa_pairs)} Q/A pairs")

    # Use vectordb directly if it's a BaseVDBConnection
    if isinstance(vectordb, BaseVDBConnection):
        vdb = vectordb
    elif query_fn is not None:
        # Fallback: create wrapper for custom functions
        class CustomVDB(BaseVDBConnection):
            def search(self, query=None, embedding=None, limit=5):
                if query:
                    results = query_fn(query, limit)
                    # Normalize results
                    normalized = []
                    for result in results:
                        if isinstance(result, dict):
                            normalized.append({
                                "text": result.get("text", ""),
                                "metadata": result.get("metadata", {}),
                                "score": result.get("score", result.get("similarity", None)),
                                "id": result.get("id", None),
                            })
                        else:
                            normalized.append({"text": str(result), "metadata": {}, "score": None})
                    return normalized
                return []
            def sample_chunks(self, num_chunks):
                if sample_fn:
                    results = sample_fn(num_chunks)
                    # Normalize results
                    normalized = []
                    for result in results:
                        if isinstance(result, dict):
                            normalized.append({
                                "text": result.get("text", ""),
                                "metadata": result.get("metadata", {}),
                                "id": result.get("id", None),
                            })
                        else:
                            normalized.append({"text": str(result), "metadata": {}})
                    return normalized
                return []
        vdb = CustomVDB()
    elif hasattr(vectordb, "query"):
        vdb = vectordb
    else:
        raise ValueError("vectordb must be a BaseVDBConnection instance or have query method")

    # Query vector DB for each question
    logger.info(f"Querying vector database with top_k={top_k}...")
    retrieval_results = []
    for qa_pair in tqdm(qa_pairs, desc="Querying vector DB", unit="query"):
        question = qa_pair.get("question", "")
        if not question:
            retrieval_results.append([])
            continue

        retrieved = vdb.query(question, top_k=top_k)
        retrieval_results.append(retrieved)

    logger.info("Calculating metrics...")
    # Calculate metrics
    metrics_result = calculate_retrieval_metrics_full(
        qa_pairs, retrieval_results, top_k=top_k
    )

    # Flatten aggregated metrics for easier access
    aggregated = metrics_result["aggregated"]
    
    # Return flattened structure for convenience
    result = {
        **aggregated,  # Flattened metrics
        "aggregated": aggregated,  # Nested structure
        "per_sample": metrics_result["per_sample"],
    }

    return result


def evaluate_rag(
    qa_dataset: Union[str, Path, List[Dict[str, Any]]],
    vectordb: Union[Any, BaseVDBConnection],
    rag_pipeline: Callable[[str], str],
    top_k: int = 5,
    query_fn: Optional[Callable] = None,
    sample_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Evaluate full RAG system (retrieval + generation).

    Args:
        qa_dataset: Path to JSONL file or list of Q/A dictionaries
        vectordb: Vector database instance (BaseVDBConnection) or custom object
        rag_pipeline: Function that takes a question and returns an answer
        top_k: Number of top results to retrieve for evaluation
        query_fn: Optional query function if using custom vector DB (deprecated)
        sample_fn: Optional sample function if using custom vector DB (deprecated)

    Returns:
        Dictionary with retrieval metrics (generation metrics deferred)
        
    Raises:
        ValidationError: If parameters or QA dataset structure is invalid
    """
    # Validate parameters
    _validate_evaluate_params(qa_dataset, top_k)
    
    # Validate RAG pipeline
    if not callable(rag_pipeline):
        raise ValidationError("rag_pipeline must be a callable function")
    
    # Load Q/A dataset
    if isinstance(qa_dataset, (str, Path)):
        qa_dataset_path = Path(qa_dataset)
        qa_pairs = []
        with open(qa_dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        qa_pair = json.loads(line)
                        _validate_qa_pair(qa_pair, i)
                        qa_pairs.append(qa_pair)
                    except json.JSONDecodeError as e:
                        raise ValidationError(f"Invalid JSON in QA dataset at line {i+1}: {e}")
    else:
        qa_pairs = qa_dataset
        # Validate all QA pairs
        for i, qa_pair in enumerate(qa_pairs):
            _validate_qa_pair(qa_pair, i)

    if not qa_pairs:
        raise ValidationError("QA dataset is empty")

    logger.info(f"Loaded {len(qa_pairs)} Q/A pairs")

    # Use vectordb directly if it's a BaseVDBConnection
    if isinstance(vectordb, BaseVDBConnection):
        vdb = vectordb
    elif query_fn is not None:
        # Fallback: create wrapper for custom functions
        class CustomVDB(BaseVDBConnection):
            def search(self, query=None, embedding=None, limit=5):
                if query:
                    results = query_fn(query, limit)
                    # Normalize results
                    normalized = []
                    for result in results:
                        if isinstance(result, dict):
                            normalized.append({
                                "text": result.get("text", ""),
                                "metadata": result.get("metadata", {}),
                                "score": result.get("score", result.get("similarity", None)),
                                "id": result.get("id", None),
                            })
                        else:
                            normalized.append({"text": str(result), "metadata": {}, "score": None})
                    return normalized
                return []
            def sample_chunks(self, num_chunks):
                if sample_fn:
                    results = sample_fn(num_chunks)
                    # Normalize results
                    normalized = []
                    for result in results:
                        if isinstance(result, dict):
                            normalized.append({
                                "text": result.get("text", ""),
                                "metadata": result.get("metadata", {}),
                                "id": result.get("id", None),
                            })
                        else:
                            normalized.append({"text": str(result), "metadata": {}})
                    return normalized
                return []
        vdb = CustomVDB()
    elif hasattr(vectordb, "query"):
        vdb = vectordb
    else:
        raise ValidationError("vectordb must be a BaseVDBConnection instance or have query method")

    # Query vector DB for each question (for retrieval evaluation)
    logger.info(f"Evaluating retrieval with top_k={top_k}...")
    retrieval_results = []
    for qa_pair in tqdm(qa_pairs, desc="Evaluating retrieval", unit="query"):
        question = qa_pair.get("question", "")
        if not question:
            retrieval_results.append([])
            continue

        retrieved = vdb.query(question, top_k=top_k)
        retrieval_results.append(retrieved)

    # Calculate retrieval metrics
    logger.info("Calculating retrieval metrics...")
    metrics_result = calculate_retrieval_metrics_full(
        qa_pairs, retrieval_results, top_k=top_k
    )

    # Evaluate generation (placeholder for future work)
    # Generation evaluation will use CRC-0.5B, GJ-0.5B, ASM-0.5B models
    logger.info("Generation evaluation: deferred (future work with CRC-0.5B, GJ-0.5B, ASM-0.5B)")

    aggregated = metrics_result["aggregated"]

    # Return structured results
    result = {
        "retrieval": {
            **aggregated,  # Flattened metrics
            "aggregated": aggregated,  # Nested structure
            "per_sample": metrics_result["per_sample"],
        },
        "generation": {
            "note": "Generation evaluation models (CRC-0.5B, GJ-0.5B, ASM-0.5B) are incoming",
        },
    }

    return result

