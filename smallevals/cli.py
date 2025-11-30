"""Command-line interface for SmallEval - Question Generation from Documents."""

import argparse
import sys
from pathlib import Path

from smallevals.generation import generate_questions_from_docs
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SmallEval - Generate questions from documents using QAG-0.5B model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 questions from documents
  smallevals --docs-dir ./documents

  # Generate 1 question per document
  smallevals --docs-dir ./documents --num-questions -1

  # Generate 3 questions per document
  smallevals --docs-dir ./documents --questions-per-doc 3

  # Specify custom output directory and chunk size
  smallevals --docs-dir ./documents --output-dir ./my_questions --min-max-chunks 1000 1500
        """
    )
    
    parser.add_argument(
        "--docs-dir",
        type=str,
        required=True,
        help="Path to directory containing documents"
    )
    
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of questions to generate (default: 100, use -1 for 1 per document)"
    )
    
    parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=None,
        help="Alternative to --num-questions: specify number of questions per document"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="smallevals_questions",
        help="Output directory for CSV files (default: smallevals_questions)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="Device to use for model inference (default: auto-detect)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for question generation (default: 8)"
    )
    
    parser.add_argument(
        "--min-max-chunks",
        type=int,
        nargs=2,
        default=[800, 1200],
        metavar=("MIN", "MAX"),
        help="Minimum and maximum chunk size in characters (default: 800 1200)"
    )
    
    args = parser.parse_args()
    
    return generate_questions_command(args)


if __name__ == "__main__":
    sys.exit(main())
