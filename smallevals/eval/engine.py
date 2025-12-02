"""Main evaluation engine with three core functions."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from tqdm import tqdm
import pandas as pd

from smallevals.vdb_integrations.base import BaseVDBConnection
from smallevals.generation.qa_generator import QAGenerator
from smallevals.eval.metrics import calculate_retrieval_metrics_full
from smallevals.exceptions import ValidationError
from smallevals.utils.logger import logger


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
    # Add VDB IDs to qa_pairs (QA generator doesn't know about VDB IDs)
    for i, (qa_pair, chunk) in enumerate(zip(qa_pairs, chunks)):
        if isinstance(chunk, dict) and "id" in chunk:
            qa_pair["id"] = chunk["id"]
    
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



def create_results_dataframe(
    qa_pairs: List[Dict[str, Any]],
    vectordb: BaseVDBConnection,
    retrieval_results: List[List[Dict[str, Any]]],
    top_k: int = 5
) -> "pd.DataFrame":
    """
    Create pandas DataFrame from QA pairs and retrieval results matching dash app format.
    
    Args:
        qa_pairs: List of QA pair dictionaries with question, answer, passage, chunk_id
        vectordb: Vector database connection instance
        retrieval_results: List of retrieval results (one list per QA pair)
        top_k: Number of top results that were retrieved
        
    Returns:
        pandas DataFrame with columns: chunk, chunk_id, question, answer, retrieved_docs, 
        retrieved_ids, num_retrieved, chunk_position
    """
    logger.info(f"Creating results DataFrame from {len(qa_pairs)} QA pairs...")
    
    rows = []
    for qa_pair, retrieved in zip(qa_pairs, retrieval_results):
        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer", "")
        passage = qa_pair.get("passage", "")
        chunk_id = qa_pair.get("id", "")  # Use VDB's ID
        
        if not question:
            continue
        
        # Extract retrieved docs and ids (VDB's IDs)
        retrieved_docs = [r.get("text", "") for r in retrieved]
        retrieved_ids = [r.get("id", "") for r in retrieved]
        
        # Find the position of the original chunk in retrieved results using VDB's ID
        chunk_position = None
        for pos, retrieved_id in enumerate(retrieved_ids):
            if retrieved_id == chunk_id:
                chunk_position = pos + 1  # 1-indexed position
                break
        
        # If not found by ID, try to match by text content (fallback)
        if chunk_position is None and passage:
            for pos, retrieved_doc in enumerate(retrieved_docs):
                if retrieved_doc == passage or retrieved_doc.strip() == passage.strip():
                    chunk_position = pos + 1  # 1-indexed position
                    break
        
        rows.append({
            "chunk": passage,
            "chunk_id": chunk_id,  # VDB's ID (keeping column name for compatibility)
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "retrieved_ids": retrieved_ids,
            "num_retrieved": len(retrieved_docs),
            "chunk_position": chunk_position  # Position of original chunk in retrieved results (1-indexed, None if not found)
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    return df


def evaluate_retrievals(
    connection: BaseVDBConnection,
    top_k: int = 5,
    n_chunks: int = 100,
    device: Optional[str] = None,
    results_folder: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality by generating QA pairs, evaluating, and saving all results.
    
    This is the main evaluation function that:
    1. Generates QA pairs from sampled chunks
    2. Evaluates retrieval quality
    3. Creates a results folder with all artifacts
    4. Generates HTML report
    5. Returns comprehensive result dictionary
    
    Args:
        connection: SmallEvalsVDBConnection or BaseVDBConnection instance
        top_k: Number of top results to retrieve per query
        n_chunks: Number of chunks to sample and generate QA pairs from
        device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
        results_folder: Optional path or name for results folder. If None, generates random name.
        batch_size: Batch size for QA generation
        
    Returns:
        Dictionary with:
        - config: Input parameters
        - results_path: Path to results folder
        - metrics: Evaluation metrics
        - qa_pairs_path: Path to qa_pairs.jsonl
        - results_csv_path: Path to retrieval_results.csv
        - html_report_path: Path to report.html
        - dataframe: Results DataFrame
    """
    from smallevals.utils.results_manager import (
        create_result_folder,
        save_evaluation_results
    )
    from smallevals.ui_dash.report_generator import generate_html_report
    
    logger.info("=" * 60)
    logger.info("Starting Retrieval Evaluation")
    logger.info("=" * 60)
    
    # Step 1: Generate QA pairs
    logger.info(f"Step 1: Generating QA pairs from {n_chunks} chunks...")
    qa_pairs = generate_qa_from_vectordb(
        vectordb=connection,
        num_chunks=n_chunks,
        device=device,
        batch_size=batch_size
    )
    
    if not qa_pairs:
        raise ValueError("No QA pairs generated. Check your vector database connection and chunks.")
    
    logger.info(f"✅ Generated {len(qa_pairs)} QA pairs")
    
    # Step 2: Evaluate retrieval quality
    logger.info(f"Step 2: Evaluating retrieval quality with top_k={top_k}...")
    
    # Query vector DB for each question
    retrieval_results = []
    for qa_pair in tqdm(qa_pairs, desc="Querying vector DB", unit="query"):
        question = qa_pair.get("question", "")
        if not question:
            retrieval_results.append([])
            continue
        retrieved = connection.query(question, top_k=top_k)
        retrieval_results.append(retrieved)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics_result = calculate_retrieval_metrics_full(
        qa_pairs, retrieval_results, top_k=top_k
    )
    aggregated = metrics_result["aggregated"]
    
    # Step 3: Create results folder
    if results_folder is None:
        result_path = create_result_folder()
    else:
        result_path = create_result_folder(results_folder)
    
    logger.info(f"Step 3: Saving results to {result_path}")
    
    # Step 4: Create DataFrame
    logger.info("Step 4: Creating results DataFrame...")
    results_df = create_results_dataframe(
        qa_pairs=qa_pairs,
        vectordb=connection,
        retrieval_results=retrieval_results,
        top_k=top_k
    )
    
    # Step 5: Prepare config
    config = {
        "top_k": top_k,
        "n_chunks": n_chunks,
        "device": device or "auto-detected",
        "batch_size": batch_size,
        "num_qa_pairs": len(qa_pairs),
    }
    
    # Get VDB type/name
    # Check if connection has vdb_type attribute (SmallEvalsVDBConnection wrapper)
    if hasattr(connection, 'vdb_type'):
        vdb_type = connection.vdb_type
    else:
        # Fall back to class name extraction
        vdb_type = connection.__class__.__name__.replace("Connection", "").lower()
   
    config["vector_db"] = vdb_type.lower()
    
    # Get embedding model info if available
    if hasattr(connection, 'embedding_model') and connection.embedding_model:
        try:
            model_name = getattr(connection.embedding_model, 'model_name', None)
            if model_name:
                config["embedding_model"] = model_name
        except Exception:
            pass
    
    # Step 6: Generate HTML report
    logger.info("Step 5: Generating HTML report...")
    version_metadata = {
        "selected_version": result_path.name,
        "description": f"Evaluation with top_k={top_k}, n_chunks={n_chunks}",
        **config
    }
    html_report = generate_html_report(
        df=results_df,
        metrics=aggregated,
        version_metadata=version_metadata,
        top_k=top_k
    )
    
    # Step 7: Save all artifacts
    logger.info("Step 6: Saving all artifacts...")

    save_evaluation_results(
        result_folder=result_path,
        qa_pairs=qa_pairs,
        results_df=results_df,
        metrics=aggregated,
        config=config,
        html_report=html_report,
    )
    
    # Step 8: Print completion message
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to: {result_path}")
    print(f"\nKey Metrics:")
    print(f"  Hit Rate@{top_k}: {aggregated.get(f'hit_rate@{top_k}', 0):.4f}")
    print(f"  Precision@{top_k}: {aggregated.get(f'precision@{top_k}', 0):.4f}")
    print(f"  Recall@{top_k}: {aggregated.get(f'recall@{top_k}', 0):.4f}")
    print(f"  MRR: {aggregated.get('mrr', 0):.4f}")
    print("\n" + "=" * 60)
    print("Run 'uv run python -m smallevals.ui_dash.app' to see results.")
    print("=" * 60 + "\n")
    
    # Return comprehensive result dictionary
    return {
        "config": config,
        "results_path": str(result_path),
        "metrics": aggregated,
        "qa_pairs_path": str(result_path / "qa_pairs.jsonl"),
        "results_csv_path": str(result_path / "retrieval_results.csv"),
        "html_report_path": str(result_path / "report.html"),
        "dataframe": results_df,
        "num_qa_pairs": len(qa_pairs),
    }
