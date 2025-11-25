"""Data organization utilities for questions and results folders."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


QUESTIONS_DIR = Path("questions")
RESULTS_DIR = Path("results")


def ensure_questions_dir():
    """Ensure the questions directory exists."""
    QUESTIONS_DIR.mkdir(exist_ok=True)


def ensure_results_dir():
    """Ensure the results directory exists."""
    RESULTS_DIR.mkdir(exist_ok=True)


def save_questions_jsonl(
    questions: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    version_name: Optional[str] = None
) -> Path:
    """
    Save questions to a JSONL file.
    
    Args:
        questions: List of question dictionaries with keys: question, answer, chunk_id, passage
        output_path: Optional custom output path
        version_name: Optional version name (used if output_path not provided)
        
    Returns:
        Path to the saved JSONL file
    """
    ensure_questions_dir()
    
    if output_path is None:
        if version_name:
            filename = f"questions_{version_name}.jsonl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"questions_{timestamp}.jsonl"
        output_path = QUESTIONS_DIR / filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for qa in questions:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    
    return output_path


def create_results_folder(
    version_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Create a results folder for a version.
    
    Args:
        version_name: Name of the version
        config: Optional initial config dictionary
        
    Returns:
        Path to the created results folder
    """
    ensure_results_dir()
    
    results_path = RESULTS_DIR / version_name
    results_path.mkdir(parents=True, exist_ok=True)
    
    if config is None:
        config = {}
    
    # Ensure created_at is set
    if "created_at" not in config:
        config["created_at"] = datetime.now().isoformat()
    
    # Save initial config
    save_results_config(results_path, config)
    
    return results_path


def save_results_config(
    results_path: Path,
    config: Dict[str, Any]
) -> None:
    """
    Save or update config.json in a results folder.
    
    Args:
        results_path: Path to the results folder
        config: Config dictionary to save
    """
    config_path = results_path / "config.json"
    
    # Load existing config if it exists
    existing_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            existing_config = json.load(f)
    
    # Merge with new config
    existing_config.update(config)
    existing_config["updated_at"] = datetime.now().isoformat()
    
    # Save
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2, ensure_ascii=False)


def load_results_config(
    results_path: Path
) -> Dict[str, Any]:
    """
    Load config.json from a results folder.
    
    Args:
        results_path: Path to the results folder
        
    Returns:
        Config dictionary
    """
    config_path = results_path / "config.json"
    
    if not config_path.exists():
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_question_file_path(
    question_file: str,
    questions_dir: Optional[Path] = None
) -> Path:
    """
    Get the full path to a question file.
    
    Args:
        question_file: Name or relative path of the question file
        questions_dir: Optional custom questions directory
        
    Returns:
        Full path to the question file
    """
    if questions_dir is None:
        questions_dir = QUESTIONS_DIR
    
    # If it's already a full path, return it
    question_path = Path(question_file)
    if question_path.is_absolute():
        return question_path
    
    # Otherwise, look in questions directory
    return questions_dir / question_file

