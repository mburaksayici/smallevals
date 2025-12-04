"""Command-line interface for smallevals.

Provides:
- ``generate_qa``: generate questions from documents.
- ``dash``: launch the Dash UI for exploring evaluation results.
"""

import argparse
import sys

from smallevals.generation import generate_questions_from_docs
from smallevals.ui_dash.app import run_dash
from smallevals.utils.logger import logger


def generate_questions_command(args):
    """CLI command for generating questions from documents."""
    try:
        logger.info("Starting question generation from documents...")
        logger.info(f"Documents directory: {args.docs_dir}")
        logger.info(f"Number of questions: {args.num_questions}")
        if args.questions_per_doc:
            logger.info(f"Questions per document: {args.questions_per_doc}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Chunk size range: {args.min_max_chunks[0]}-{args.min_max_chunks[1]} characters")
        
        csv_path = generate_questions_from_docs(
            docs_path=args.docs_dir,
            num_questions=args.num_questions,
            questions_per_doc=args.questions_per_doc,
            min_max_chunks=args.min_max_chunks,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        
        print(f"\n✅ Successfully generated questions!")
        print(f"📄 Output file: {csv_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


def dash_command(args):
    """CLI command for running the Dash UI."""
    logger.info("Starting smallevals Dash UI...")
    logger.info(f"Host: {args.host}, Port: {args.port}, Debug: {args.debug}")
    run_dash(host=args.host, port=args.port, debug=args.debug)
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="smallevals - Small Language Models Evaluation Suite for RAG Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 questions from documents
  smallevals generate_qa --docs-dir ./documents

  # Generate 1 question per document
  smallevals generate_qa --docs-dir ./documents --num-questions -1

  # Generate 3 questions per document
  smallevals generate_qa --docs-dir ./documents --questions-per-doc 3

  # Specify custom output directory and chunk size
  smallevals generate_qa --docs-dir ./documents --output-dir ./my_questions --min-max-chunks 1000 1500

  # Run the Dash UI
  smallevals dash --host 0.0.0.0 --port 8050 --debug
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate_qa subcommand
    generate_parser = subparsers.add_parser(
        "generate_qa",
        help="Generate QA from documents using the QAG-0.6B model",
    )
    generate_parser.add_argument(
        "--docs-dir",
        type=str,
        required=True,
        help="Path to directory containing documents",
    )
    generate_parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of questions to generate (default: 100, use -1 for 1 per document)",
    )
    generate_parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=None,
        help="Alternative to --num-questions: specify number of questions per document",
    )
    generate_parser.add_argument(
        "--output-dir",
        type=str,
        default="smallevals_questions",
        help="Output directory for CSV files (default: smallevals_questions)",
    )
    generate_parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="Device to use for model inference (default: auto-detect)",
    )
    generate_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for question generation (default: 8)",
    )
    generate_parser.add_argument(
        "--min-max-chunks",
        type=int,
        nargs=2,
        default=[800, 1200],
        metavar=("MIN", "MAX"),
        help="Minimum and maximum chunk size in characters (default: 800 1200)",
    )
    generate_parser.set_defaults(func=generate_questions_command)

    # dash subcommand
    dash_parser = subparsers.add_parser(
        "dash",
        help="Launch the smallevals Dash UI",
    )
    dash_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind (default: 0.0.0.0)",
    )
    dash_parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port number (default: 8050)",
    )
    dash_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode",
    )
    dash_parser.set_defaults(func=dash_command)

    args = parser.parse_args()

    # Dispatch to the chosen subcommand
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
