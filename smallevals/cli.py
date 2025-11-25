"""Command-line interface for SmallEval."""

import argparse
import json
from pathlib import Path
from typing import Optional

from smallevals.api import (
    generate_qa_from_vectordb,
    evaluate_vectordb,
    evaluate_rag,
)


def generate_qa_command(args):
    """CLI command for generating QA from vector DB."""
    print(f"Generating QA pairs from vector database...")
    print(f"Vector DB: {args.vectordb}")
    print(f"Number of chunks: {args.num_chunks}")
    print(f"Output: {args.output}")
    print(f"Model: Using hardcoded model from HuggingFace")
    
    # Note: In a real implementation, you would load the vector DB here
    # For now, this is a placeholder structure
    print("Note: Vector DB connection needs to be configured in code")
    print("Use the Python API directly for full functionality")


def evaluate_vectordb_command(args):
    """CLI command for evaluating vector DB."""
    print(f"Evaluating vector database...")
    print(f"QA Dataset: {args.qa_dataset}")
    print(f"Vector DB: {args.vectordb}")
    print(f"Top K: {args.top_k}")
    
    # Note: In a real implementation, you would load the vector DB here
    print("Note: Vector DB connection needs to be configured in code")
    print("Use the Python API directly for full functionality")


def evaluate_rag_command(args):
    """CLI command for evaluating RAG system."""
    print(f"Evaluating RAG system...")
    print(f"QA Dataset: {args.qa_dataset}")
    print(f"Vector DB: {args.vectordb}")
    print(f"Top K: {args.top_k}")
    
    # Note: In a real implementation, you would load the vector DB and RAG pipeline here
    print("Note: Vector DB connection and RAG pipeline need to be configured in code")
    print("Use the Python API directly for full functionality")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SmallEval - Small Language Models Evaluation Suite for RAG Systems"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate QA command
    gen_parser = subparsers.add_parser(
        "generate-qa",
        help="Generate Q/A pairs from vector database"
    )
    gen_parser.add_argument(
        "--vectordb",
        type=str,
        required=True,
        help="Vector database configuration or connection string"
    )
    gen_parser.add_argument(
        "--num-chunks",
        type=int,
        default=100,
        help="Number of chunks to sample (default: 100)"
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    gen_parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        help="Device to use (auto-detect if not specified)"
    )
    gen_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)"
    )
    gen_parser.set_defaults(func=generate_qa_command)
    
    # Evaluate vector DB command
    eval_vdb_parser = subparsers.add_parser(
        "evaluate-vectordb",
        help="Evaluate vector database retrieval quality"
    )
    eval_vdb_parser.add_argument(
        "--qa-dataset",
        type=str,
        required=True,
        help="Path to JSONL file with Q/A pairs"
    )
    eval_vdb_parser.add_argument(
        "--vectordb",
        type=str,
        required=True,
        help="Vector database configuration or connection string"
    )
    eval_vdb_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5)"
    )
    eval_vdb_parser.set_defaults(func=evaluate_vectordb_command)
    
    # Evaluate RAG command
    eval_rag_parser = subparsers.add_parser(
        "evaluate-rag",
        help="Evaluate full RAG system"
    )
    eval_rag_parser.add_argument(
        "--qa-dataset",
        type=str,
        required=True,
        help="Path to JSONL file with Q/A pairs"
    )
    eval_rag_parser.add_argument(
        "--vectordb",
        type=str,
        required=True,
        help="Vector database configuration or connection string"
    )
    eval_rag_parser.add_argument(
        "--rag-pipeline",
        type=str,
        help="Path to Python file with RAG pipeline function (optional)"
    )
    eval_rag_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5)"
    )
    eval_rag_parser.set_defaults(func=evaluate_rag_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
