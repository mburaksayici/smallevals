"""Version management system for storing and loading evaluation results with different embeddings."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


VERSIONS_DIR = Path("versions")


def ensure_versions_dir():
    """Ensure the versions directory exists."""
    VERSIONS_DIR.mkdir(exist_ok=True)


def create_version(
    version_name: str,
    description: str = "",
    embedding_model: str = "",
    top_k: int = 5
) -> Path:
    """
    Create a new version folder.
    
    Args:
        version_name: Name of the version (used as folder name)
        description: Description of this version
        embedding_model: Name of the embedding model used
        top_k: Top-k value used for retrieval
        
    Returns:
        Path to the created version directory
    """
    ensure_versions_dir()
    
    version_path = VERSIONS_DIR / version_name
    version_path.mkdir(exist_ok=True)
    
    # Create metadata.json
    metadata = {
        "version_name": version_name,
        "embedding_model": embedding_model,
        "created_at": datetime.now().isoformat(),
        "description": description,
        "top_k": top_k
    }
    
    metadata_path = version_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return version_path


def list_versions() -> List[str]:
    """
    List all available versions.
    
    Returns:
        List of version names (folder names)
    """
    ensure_versions_dir()
    
    if not VERSIONS_DIR.exists():
        return []
    
    versions = []
    for item in VERSIONS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Check if it has metadata.json (valid version folder)
            if (item / "metadata.json").exists():
                versions.append(item.name)
    
    return sorted(versions)


def get_version_metadata(version_name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a version.
    
    Args:
        version_name: Name of the version
        
    Returns:
        Dictionary with metadata or None if version doesn't exist
    """
    version_path = VERSIONS_DIR / version_name
    metadata_path = version_path / "metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_version(version_name: str) -> Dict[str, Any]:
    """
    Load all data from a version folder.
    
    Args:
        version_name: Name of the version to load
        
    Returns:
        Dictionary with keys: 'qa_pairs', 'chunks', 'results_df', 'metadata'
    """
    version_path = VERSIONS_DIR / version_name
    
    if not version_path.exists():
        raise ValueError(f"Version '{version_name}' does not exist")
    
    result = {}
    
    # Load metadata
    metadata_path = version_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            result["metadata"] = json.load(f)
    else:
        result["metadata"] = {}
    
    # Load QA pairs
    qa_path = version_path / "qa_pairs.jsonl"
    qa_pairs = []
    if qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))
    result["qa_pairs"] = qa_pairs
    
    # Load chunks
    chunks_path = version_path / "chunks.jsonl"
    chunks = []
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
    result["chunks"] = chunks
    
    # Load retrieval results CSV
    results_path = version_path / "retrieval_results.csv"
    result["results_df"] = None
    if results_path.exists():
        try:
            import pandas as pd
            result["results_df"] = pd.read_csv(results_path)
        except ImportError:
            # Fallback to manual CSV reading if pandas not available
            with open(results_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                result["results_df"] = list(reader)
    
    return result


def save_to_version(
    version_name: str,
    qa_pairs: Optional[List[Dict[str, Any]]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    results_df: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save data to a version folder.
    
    Args:
        version_name: Name of the version (will create if doesn't exist)
        qa_pairs: List of QA pair dictionaries
        chunks: List of chunk dictionaries
        results_df: Pandas DataFrame or list of dicts with retrieval results
        metadata: Optional metadata to update
        
    Returns:
        Path to the version directory
    """
    ensure_versions_dir()
    version_path = VERSIONS_DIR / version_name
    version_path.mkdir(exist_ok=True)
    
    # Update metadata if provided
    if metadata:
        metadata_path = version_path / "metadata.json"
        existing_metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        existing_metadata.update(metadata)
        existing_metadata["updated_at"] = datetime.now().isoformat()
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
    
    # Save QA pairs
    if qa_pairs is not None:
        qa_path = version_path / "qa_pairs.jsonl"
        with open(qa_path, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                # Add version to each QA pair
                qa_with_version = qa.copy()
                qa_with_version["version"] = version_name
                f.write(json.dumps(qa_with_version, ensure_ascii=False) + "\n")
    
    # Save chunks
    if chunks is not None:
        chunks_path = version_path / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                chunk_with_version = chunk.copy()
                chunk_with_version["version"] = version_name
                f.write(json.dumps(chunk_with_version, ensure_ascii=False) + "\n")
    
    # Save retrieval results
    if results_df is not None:
        results_path = version_path / "retrieval_results.csv"
        try:
            import pandas as pd
            if isinstance(results_df, pd.DataFrame):
                results_df.to_csv(results_path, index=False)
            else:
                # Convert list of dicts to DataFrame
                df = pd.DataFrame(results_df)
                df.to_csv(results_path, index=False)
        except ImportError:
            # Fallback to manual CSV writing
            if results_df and len(results_df) > 0:
                fieldnames = results_df[0].keys() if isinstance(results_df, list) else results_df.columns
                with open(results_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    if isinstance(results_df, list):
                        writer.writerows(results_df)
                    else:
                        for _, row in results_df.iterrows():
                            writer.writerow(row.to_dict())
    
    return version_path

