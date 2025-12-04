"""Main evaluation engine with three core functions."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from tqdm import tqdm
import pandas as pd
import ast

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
            def search(self, query=None, embedding=None, top_k=5):
                if query:
                    results = query_fn(query, top_k)
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
                    return query_fn(None, top_k)  # This may need adjustment
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
    elif hasattr(vectordb, "search") and hasattr(vectordb, "sample_chunks"):
        # Use object directly if it has the right methods
        vdb = vectordb
    else:
        raise ValueError("vectordb must be a BaseVDBConnection instance or have search/sample_chunks methods")
    # Sample chunks from vector DB
    logger.info(f"Sampling {num_chunks} chunks from vector database...")
    chunks = vdb.sample_chunks(num_chunks)

    logger.info(f"Sampled {len(chunks)} chunks")

    # Generate Q/A pairs (uses hardcoded model from HuggingFace)
    logger.info("Generating Q/A pairs...")
    qa_generator = QAGenerator(device=device, batch_size=batch_size)

    qa_pairs = qa_generator.generate_from_chunks(chunks, max_retries=1)
    
    # NOTE: qa_pairs may be shorter than chunks if some generations failed
    # We need to match qa_pairs back to their original chunks by text content
    
    # Create a mapping of passage text to chunk for ID assignment
    chunk_map = {}
    for chunk in chunks:
        if isinstance(chunk, dict) and "text" in chunk:
            # Use text as key to find the original chunk
            chunk_map[chunk["text"]] = chunk
    
    # Add VDB IDs to qa_pairs by matching passage text
    for qa_pair in qa_pairs:
        passage_text = qa_pair.get("passage", "")
        if passage_text in chunk_map:
            chunk = chunk_map[passage_text]
            if "id" in chunk:
                qa_pair["id"] = chunk["id"]
    
    logger.info(f"Generated {len(qa_pairs)} Q/A pairs (from {len(chunks)} chunks)")

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
            "chunk_id": chunk_id,  
            "chunk_position": chunk_position,# Position of original chunk in retrieved results (1-indexed, None if not found)
            "retrieved_ids": retrieved_ids,
            "num_retrieved": len(retrieved_docs),
            "question": question,
            "answer": answer,
            "chunk": passage,
            "retrieved_docs": retrieved_docs,
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
        retrieved = connection.search(question, top_k=top_k)
        retrieval_results.append(retrieved)
    
    # Calculate metrics for main top_k
    logger.info("Calculating metrics...")
    metrics_result = calculate_retrieval_metrics_full(
        qa_pairs, retrieval_results, top_k=top_k
    )
    aggregated = metrics_result["aggregated"]
    
    # Also calculate metrics for top_k=1
    logger.info("Calculating Top-1 metrics...")
    metrics_result_top1 = calculate_retrieval_metrics_full(
        qa_pairs, retrieval_results, top_k=1
    )
    aggregated_top1 = metrics_result_top1["aggregated"]
    
    # Merge top-1 metrics into aggregated metrics
    aggregated.update(aggregated_top1)
    
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


def recalculate_metrics_from_eval_folder(
    eval_folder: Union[str, Path],
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Recalculate metrics and regenerate reports from an existing evaluation folder.
    
    This function is useful for fixing metrics that were calculated incorrectly
    in a previous version of the code, or for regenerating reports with updated
    templates.
    
    Args:
        eval_folder: Path to the evaluation folder (e.g., "smallevals_results/eval_4a290bad")
        overwrite: Whether to overwrite existing metrics and report files
        
    Returns:
        Dictionary with:
        - config: Original configuration
        - results_path: Path to results folder
        - metrics: Recalculated metrics
        - old_metrics: Original (incorrect) metrics for comparison
        - qa_pairs_path: Path to qa_pairs.jsonl
        - results_csv_path: Path to retrieval_results.csv
        - html_report_path: Path to updated report.html
        
    Raises:
        ValueError: If eval folder doesn't exist or is missing required files
    """
    from smallevals.ui_dash.report_generator import generate_html_report
    from smallevals.utils.results_manager import save_evaluation_results
    
    eval_path = Path(eval_folder)
    
    # Validate folder exists
    if not eval_path.exists():
        raise ValueError(f"Evaluation folder does not exist: {eval_path}")
    
    logger.info("=" * 60)
    logger.info(f"Recalculating Metrics for: {eval_path.name}")
    logger.info("=" * 60)
    
    # Load existing files
    qa_pairs_path = eval_path / "qa_pairs.jsonl"
    results_csv_path = eval_path / "retrieval_results.csv"
    metadata_path = eval_path / "metadata.json"
    
    # Validate required files exist
    if not qa_pairs_path.exists():
        raise ValueError(f"Missing qa_pairs.jsonl in {eval_path}")
    if not results_csv_path.exists():
        raise ValueError(f"Missing retrieval_results.csv in {eval_path}")
    
    # Load QA pairs
    logger.info("Loading QA pairs...")
    qa_pairs = []
    with open(qa_pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Load results CSV
    logger.info("Loading retrieval results...")
    results_df = pd.read_csv(results_csv_path)
    logger.info(f"Loaded {len(results_df)} retrieval results")
    
    # Load metadata (if exists)
    old_metrics = {}
    config = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            old_metrics = metadata.get("metrics", {})
            config = metadata.get("config", {})
    
    # Extract top_k from config or infer from data
    top_k = config.get("top_k", 10)
    
    # Reconstruct retrieval_results from CSV
    logger.info("Reconstructing retrieval results from CSV...")
    retrieval_results = []
    
    for idx, row in results_df.iterrows():
        # Parse retrieved_docs and retrieved_ids from CSV
        # These are stored as string representations of lists
        retrieved_docs_str = row.get("retrieved_docs", "[]")
        retrieved_ids_str = row.get("retrieved_ids", "[]")
        
        try:
            # Use ast.literal_eval to safely parse the list strings
            retrieved_docs = ast.literal_eval(retrieved_docs_str)
            retrieved_ids = ast.literal_eval(retrieved_ids_str)
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Failed to parse row {idx}: {e}")
            retrieval_results.append([])
            continue
        
        # Reconstruct list of dicts matching the format expected by metrics
        retrieved = []
        for doc, doc_id in zip(retrieved_docs, retrieved_ids):
            retrieved.append({
                "text": doc,
                "id": doc_id,
            })
        
        retrieval_results.append(retrieved)
    
    logger.info(f"Reconstructed {len(retrieval_results)} retrieval result sets")
    
    # Recalculate metrics for main top_k
    logger.info("Recalculating metrics...")
    metrics_result = calculate_retrieval_metrics_full(
        qa_pairs, retrieval_results, top_k=top_k
    )
    aggregated = metrics_result["aggregated"]
    
    # Also calculate metrics for top_k=1
    logger.info("Calculating Top-1 metrics...")
    metrics_result_top1 = calculate_retrieval_metrics_full(
        qa_pairs, retrieval_results, top_k=1
    )
    aggregated_top1 = metrics_result_top1["aggregated"]
    
    # Merge top-1 metrics into aggregated metrics
    aggregated.update(aggregated_top1)
    
    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("Metrics Comparison")
    logger.info("=" * 60)
    
    if old_metrics:
        for key in sorted(aggregated.keys()):
            old_val = old_metrics.get(key, 0.0)
            new_val = aggregated.get(key, 0.0)
            diff = new_val - old_val
            
            if abs(diff) > 0.001:
                logger.info(f"{key:20s}: {old_val:8.4f} → {new_val:8.4f} (Δ {diff:+.4f})")
            else:
                logger.info(f"{key:20s}: {new_val:8.4f} (unchanged)")
    else:
        for key, val in sorted(aggregated.items()):
            logger.info(f"{key:20s}: {val:8.4f}")
    
    # Regenerate HTML report
    logger.info("\nRegenerating HTML report...")
    version_metadata = {
        "selected_version": eval_path.name,
        "description": f"Recalculated metrics for evaluation with top_k={top_k}",
        **config
    }
    html_report = generate_html_report(
        df=results_df,
        metrics=aggregated,
        version_metadata=version_metadata,
        top_k=top_k
    )
    
    # Save updated results if overwrite is True
    if overwrite:
        logger.info("Saving updated metrics and report...")
        
        # Update metadata
        updated_metadata = {
            "created_at": metadata.get("created_at") if metadata_path.exists() else None,
            "recalculated_at": pd.Timestamp.now().isoformat(),
            "config": config,
            "metrics": aggregated,
            "old_metrics": old_metrics,  # Keep for reference
        }
        
        # Save updated metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(updated_metadata, f, indent=2, ensure_ascii=False)
        
        # Save updated metrics
        metrics_path = eval_path / "evaluation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        
        # Save updated HTML report
        report_path = eval_path / "report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        
        logger.info(f"✅ Updated files saved to: {eval_path}")
    else:
        logger.info("Skipping save (overwrite=False)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Metrics Recalculation Complete!")
    print("=" * 60)
    print(f"Folder: {eval_path}")
    print(f"\nRecalculated Metrics:")
    print(f"  Hit Rate@{top_k}: {aggregated.get(f'hit_rate@{top_k}', 0):.4f}")
    print(f"  Hit Rate@1: {aggregated.get('hit_rate@1', 0):.4f}")
    print(f"  Precision@{top_k}: {aggregated.get(f'precision@{top_k}', 0):.4f}")
    print(f"  Recall@{top_k}: {aggregated.get(f'recall@{top_k}', 0):.4f}")
    print(f"  Recall@1: {aggregated.get('recall@1', 0):.4f}")
    print(f"  MRR: {aggregated.get('mrr', 0):.4f}")
    print(f"  nDCG@{top_k}: {aggregated.get(f'ndcg@{top_k}', 0):.4f}")
    print("=" * 60 + "\n")
    
    return {
        "config": config,
        "results_path": str(eval_path),
        "metrics": aggregated,
        "old_metrics": old_metrics,
        "qa_pairs_path": str(qa_pairs_path),
        "results_csv_path": str(results_csv_path),
        "html_report_path": str(eval_path / "report.html"),
        "dataframe": results_df,
        "num_qa_pairs": len(qa_pairs),
    }
