"""Results management system for storing and loading evaluation results in smallevals_results folder."""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

import pandas as pd

RESULTS_DIR = Path("smallevals_results")


def ensure_results_dir():
    """Ensure the smallevals_results directory exists."""
    RESULTS_DIR.mkdir(exist_ok=True)


def create_result_folder(result_name: Optional[str] = None) -> Path:
    """
    Create a new result folder with a random name or specified name.
    
    Args:
        result_name: Optional name for the result folder. If None, generates a random UUID.
        
    Returns:
        Path to the created result directory
    """
    ensure_results_dir()
    
    if result_name is None:
        # Generate random name
        result_name = f"eval_{uuid.uuid4().hex[:8]}"
    
    result_path = RESULTS_DIR / result_name
    result_path.mkdir(exist_ok=True, parents=True)
    
    return result_path


def save_evaluation_results(
    result_folder: Union[str, Path],
    qa_pairs: Optional[List[Dict[str, Any]]] = None,
    results_df: Optional[pd.DataFrame] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    html_report: Optional[str] = None,
) -> Path:
    """
    Save all evaluation artifacts to a result folder.
    
    Args:
        result_folder: Path to the result folder (created if doesn't exist)
        qa_pairs: List of QA pair dictionaries
        results_df: Pandas DataFrame with retrieval results
        metrics: Dictionary of aggregated metrics
        config: Dictionary of configuration parameters
        html_report: HTML report string
        
    Returns:
        Path to the result directory
    """
    result_path = Path(result_folder)
    result_path.mkdir(exist_ok=True, parents=True)
    
    # Prepare metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "config": config or {},
        "metrics": metrics or {},
    }
    
    # Save metadata.json
    metadata_path = result_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save QA pairs
    if qa_pairs is not None:
        qa_path = result_path / "qa_pairs.jsonl"
        with open(qa_path, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                # Ensure VDB's ID is stored as string
                if "id" in qa:
                    qa["id"] = str(qa["id"])
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    
    # Save retrieval results CSV
    if results_df is not None:
        results_path = result_path / "retrieval_results.csv"
        results_df.to_csv(results_path, index=False)
    
    # Save metrics as JSON
    if metrics is not None:
        metrics_path = result_path / "evaluation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Save HTML report
    if html_report is not None:
        report_path = result_path / "report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_report)
    
    return result_path


def list_results() -> List[str]:
    """
    List all available result folders.
    
    Returns:
        List of result folder names (sorted by creation time, newest first)
    """
    ensure_results_dir()
    
    if not RESULTS_DIR.exists():
        return []
    
    results = []
    for item in RESULTS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Check if it has metadata.json (valid result folder)
            if (item / "metadata.json").exists():
                results.append(item.name)
    
    # Sort by creation time (newest first)
    def get_creation_time(name: str) -> float:
        metadata_path = RESULTS_DIR / name / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    created_at = metadata.get("created_at", "")
                    if created_at:
                        return datetime.fromisoformat(created_at).timestamp()
            except Exception:
                pass
        # Fall back to file modification time
        try:
            return (RESULTS_DIR / name / "metadata.json").stat().st_mtime
        except Exception:
            return 0.0
    
    return sorted(results, key=get_creation_time, reverse=True)


def get_result_metadata(result_name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a result folder.
    
    Args:
        result_name: Name of the result folder
        
    Returns:
        Dictionary with metadata or None if result doesn't exist
    """
    result_path = RESULTS_DIR / result_name
    metadata_path = result_path / "metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_result(result_name: str) -> Dict[str, Any]:
    """
    Load all data from a result folder.
    
    Args:
        result_name: Name of the result folder to load
        
    Returns:
        Dictionary with keys: 'qa_pairs', 'results_df', 'metadata', 'metrics', 'html_report_path'
    """
    result_path = RESULTS_DIR / result_name
    
    if not result_path.exists():
        raise ValueError(f"Result folder '{result_name}' does not exist")
    
    result = {}
    
    # Load metadata
    metadata_path = result_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            result["metadata"] = metadata
            result["config"] = metadata.get("config", {})
            result["metrics"] = metadata.get("metrics", {})
    else:
        result["metadata"] = {}
        result["config"] = {}
        result["metrics"] = {}
    
    # Load QA pairs
    qa_path = result_path / "qa_pairs.jsonl"
    qa_pairs = []
    if qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))
    result["qa_pairs"] = qa_pairs
    
    # Load retrieval results CSV
    results_path = result_path / "retrieval_results.csv"
    result["results_df"] = None
    if results_path.exists():
        result["results_df"] = pd.read_csv(results_path)
    
    # Store paths
    result["result_path"] = result_path
    result["qa_pairs_path"] = qa_path if qa_path.exists() else None
    result["results_csv_path"] = results_path if results_path.exists() else None
    result["html_report_path"] = result_path / "report.html" if (result_path / "report.html").exists() else None
    
    return result

