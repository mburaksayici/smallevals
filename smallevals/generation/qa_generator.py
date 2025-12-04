"""QA generation from chunks using exact prompt format."""

import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import pandas as pd
import os
import nltk

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from smallevals.models import GoldenGenerator
from smallevals.utils.json_parser import parse_json_response
from smallevals.exceptions import ValidationError, QAGenerationError
from smallevals.utils.logger import logger


def _ensure_nltk_data():
    """Ensure required NLTK data is available."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab', quiet=True)


def _extract_sentence_aware_chunk(
    content: str,
    start_pos: int,
    min_chunk_size: int,
    max_chunk_size: int
) -> str:
    """
    Extract a chunk that respects sentence boundaries when possible.
    
    Args:
        content: Full text content
        start_pos: Starting position in content
        min_chunk_size: Minimum desired chunk size
        max_chunk_size: Maximum desired chunk size
        
    Returns:
        Extracted chunk text
    """
    # Calculate available content from start position
    remaining_content = content[start_pos:]
    available_length = len(remaining_content)
    
    # Edge case: if remaining content is less than min_chunk_size, just return it all
    if available_length <= min_chunk_size:
        return remaining_content.strip()
    
    # Determine target chunk size (random between min and max, but not exceeding available)
    target_size = random.randint(min_chunk_size, min(max_chunk_size, available_length))
    
    # Extract initial chunk
    raw_chunk = remaining_content[:target_size]
    
    try:
        # Find first period and trim before it to start at sentence boundary
        first_period = raw_chunk.find('.')
        if first_period > 0 and first_period < len(raw_chunk) * 0.3:  # Only trim if period is in first 30%
            raw_chunk = raw_chunk[first_period + 1:].lstrip()
        
        # Edge case: if after trimming we have very little content, skip sentence processing
        if len(raw_chunk) < min_chunk_size * 0.5:
            return remaining_content[:target_size].strip()
        
        # Use NLTK to split into sentences
        sentences = nltk.sent_tokenize(raw_chunk)
        
        # Rebuild chunk from complete sentences
        chunk_text = ""
        for sentence in sentences:
            potential_chunk = chunk_text + " " + sentence if chunk_text else sentence
            if len(potential_chunk) <= max_chunk_size:
                chunk_text = potential_chunk
            else:
                break
        
        # Ensure we have reasonable content
        if len(chunk_text) < min_chunk_size * 0.6:
            # Fall back to original chunk if sentence-aware version is too short
            return raw_chunk.strip()
        
        return chunk_text.strip()
            
    except Exception as e:
        # Fall back to non-semantic direct cut on any error
        logger.debug(f"Sentence tokenization failed, using direct cut: {e}")
        return raw_chunk.strip()


class QAGenerator:
    """Generates Q/A pairs from chunks using QAG-0.5B model."""

    PROMPT_TEMPLATE = (
        'Given the passage below, extract ONE question/answer pair grounded strictly in a single atomic fact.\n\n'
        'PASSAGE:\n"<.<passage>.>"\n'
        'Return ONLY a JSON object.'
    )

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        """
        Initialize QA generator.

        Args:
            device: Device to use ("cuda", "cpu", "mps", or None for auto-detect)
            batch_size: Batch size for generation
            
        Raises:
            ValidationError: If batch_size is invalid
        """
        if batch_size <= 0:
            raise ValidationError(f"batch_size must be positive, got {batch_size}")
        
        # Uses hardcoded model from HuggingFace (configured in GoldenGenerator)
        self.model_loader = GoldenGenerator(device=device, batch_size=batch_size)

    def format_prompt(self, passage: str) -> str:
        """
        Format prompt with passage.

        Args:
            passage: Text passage to generate Q/A from

        Returns:
            Formatted prompt string
        """
        return self.PROMPT_TEMPLATE.replace('<.<passage>.>', passage)

    def generate_qa(self, passage: str) -> Optional[Dict[str, Any]]:
        """
        Generate single Q/A pair from passage.

        Args:
            passage: Text passage

        Returns:
            Dictionary with "question", "answer", and "passage" keys, or None if generation fails
        """
        prompt = self.format_prompt(passage)
        responses = self.model_loader.generate([prompt], max_new_tokens=400, temperature=0.7)
        
        if not responses:
            return None

        response = responses[0]
        parsed = parse_json_response(response)

        if parsed and "question" in parsed and "answer" in parsed:
            return {
                "question": parsed["question"],
                "answer": parsed["answer"],
                "passage": passage,
            }

        return None

    def generate_qa_batch(
        self, passages: List[str], max_retries: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate Q/A pairs from multiple passages in batch.

        Args:
            passages: List of text passages
            max_retries: Maximum number of retries for failed generations

        Returns:
            List of dictionaries with "question", "answer", and "passage" keys
        """
        if not passages:
            return []
        
        if max_retries < 0:
            raise ValidationError(f"max_retries must be non-negative, got {max_retries}")

        # Format prompts
        prompts = [self.format_prompt(passage) for passage in passages]
        
        # Generate responses with progress bar
        logger.debug(f"Generating Q/A pairs for {len(passages)} passages")
        responses = self.model_loader.generate_batched(
            prompts, max_new_tokens=7000, temperature=0.0
        )

        # Parse responses - skip failed generations entirely
        results = []
        failed_count = 0
        
        for i, (passage, response) in enumerate(zip(passages, responses)):
            parsed = parse_json_response(response)
            
            if parsed and "question" in parsed and "answer" in parsed:
                results.append({
                    "question": parsed["question"],
                    "answer": parsed["answer"],
                    "passage": passage,
                })
            else:
                # Retry if parsing failed
                retry_success = False
                if max_retries > 0:
                    logger.debug(f"Retrying generation for passage {i}")
                    retry_result = self.generate_qa(passage)
                    if retry_result:
                        results.append(retry_result)
                        retry_success = True
                
                if not retry_success:
                    logger.warning(f"Skipping passage {i} - failed to generate QA")
                    failed_count += 1

        if failed_count > 0:
            logger.warning(f"Skipped {failed_count} passages due to failed QA generation")

        return results

    def generate_from_chunks(
        self, chunks: List[Dict[str, Any]], max_retries: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate Q/A pairs from chunks (dictionaries with "text" key).

        Args:
            chunks: List of chunk dictionaries with "text" key
            max_retries: Maximum number of retries for failed generations

        Returns:
            List of Q/A dictionaries
            
        Raises:
            ValidationError: If chunks is empty or invalid
        """
        if not chunks:
            raise ValidationError("chunks list cannot be empty")
        
        passages = []
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict) and "text" in chunk:
                passages.append(chunk["text"])
                chunk_metadata.append(chunk.get("metadata", {}))
            elif isinstance(chunk, str):
                passages.append(chunk)
                chunk_metadata.append({})
            else:
                logger.warning(f"Chunk at index {i} is not a dict with 'text' key or string, converting to string")
                passages.append(str(chunk))
                chunk_metadata.append({})

        if not passages:
            raise ValidationError("No valid passages extracted from chunks")

        qa_pairs = self.generate_qa_batch(passages, max_retries=max_retries)
        
        return qa_pairs


def generate_questions_from_docs(
    docs_path: Union[str, List[str]],
    num_questions: int = 100,
    questions_per_doc: Optional[int] = None,
    min_max_chunks: List[int] = [800, 1200],
    output_dir: str = "smallevals_questions",
    device: Optional[str] = None,
    batch_size: int = 8,
) -> str:
    """
    Generate questions from documents.
    
    Args:
        docs_path: Path to directory, single file, or list of file paths
        num_questions: Number of questions to generate. If -1, generate 1 per document.
        questions_per_doc: Alternative to num_questions, specify questions per document
        min_max_chunks: List with [min_chunk_size, max_chunk_size] (default: [800, 1200])
        output_dir: Output directory for CSV files (default: smallevals_questions)
        device: Device to use ("cuda", "cpu", "mps", or None for auto-detect)
        batch_size: Batch size for generation (default: 8)
    
    Returns:
        Path to the generated CSV file
        
    Raises:
        ValidationError: If parameters are invalid
        FileNotFoundError: If docs_path doesn't exist
    """
    min_chunk_size, max_chunk_size = min_max_chunks[0], min_max_chunks[1]
    
    # Handle different input types: directory, single file, or list of files
    text_files = []
    
    if isinstance(docs_path, str):
        path_obj = Path(docs_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {docs_path}")
        
        if path_obj.is_dir():
            # Load documents from directory using docling
            # Recursively find all files in the directory
            text_files = []
            for root, dirs, files in os.walk(path_obj):
                for filename in files:
                    file_path = Path(root) / filename
                    # Try to verify it's a readable document by checking extension
                    if file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.md', '.doc', '.pptx', '.html', ".md"]:
                        text_files.append(file_path)
            
            if not text_files:
                raise ValidationError(f"No documents found in {docs_path}")
        elif path_obj.is_file():
            # Single file - validate it exists
            text_files = [path_obj]
        else:
            raise ValidationError(f"Path is neither a directory nor a file: {docs_path}")
    
    elif isinstance(docs_path, list):
        # List of file paths
        for file_path_str in docs_path:
            file_path = Path(file_path_str)
            if not file_path.exists():
                logger.warning(f"File not found, skipping: {file_path_str}")
                continue
            if not file_path.is_file():
                logger.warning(f"Path is not a file, skipping: {file_path_str}")
                continue
            text_files.append(file_path)
        
        if not text_files:
            raise ValidationError("No valid files found in the provided list")
    
    else:
        raise ValidationError(f"docs_path must be a string or list of strings, got {type(docs_path)}")
    
    logger.info(f"Found {len(text_files)} files")
    
    # Determine question generation strategy
    if questions_per_doc is not None:
        # Generate specified number of questions per document
        target_questions = len(text_files) * questions_per_doc
        file_question_map = {str(f): questions_per_doc for f in text_files}
    elif num_questions == -1:
        # Generate 1 question per document
        target_questions = len(text_files)
        file_question_map = {str(f): 1 for f in text_files}
    else:
        # If num_questions > number of docs, ensure every doc gets 1 question first
        # Then randomly select documents for remaining questions
        file_question_map = {}
        
        # First, ensure every document gets at least 1 question
        for file_path in text_files:
            file_question_map[str(file_path)] = 1
        
        # If we need more questions, randomly distribute the remaining
        remaining = num_questions - len(text_files)
        if remaining > 0:
            while remaining > 0:
                file_path = random.choice(text_files)
                file_str = str(file_path)
                file_question_map[file_str] += 1
                remaining -= 1
        elif remaining < 0:
            # If num_questions < number of docs, randomly select which docs to use
            selected_files = random.sample(text_files, num_questions)
            file_question_map = {str(f): 1 for f in selected_files}
    
    # Ensure NLTK data is available
    _ensure_nltk_data()
    
    # Initialize QA generator
    logger.info("Initializing QA generator...")
    qa_generator = QAGenerator(device=device, batch_size=batch_size)

    # Initialize Docling converter with OCR explicitly disabled for PDFs
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )

    # Collect chunks to process
    chunks_to_process = []
    for file_path_str, num_q in file_question_map.items():
        file_path = Path(file_path_str)

        try:
            # Load file content using Docling converter for all supported formats
            result = converter.convert(str(file_path))

            # Extract plain text content
            content = result.document.export_to_text().strip()

            if not content:
                logger.warning(f"No content loaded from {file_path.name}")
                continue
        except Exception as e:
            logger.warning(f"Error loading {file_path.name}: {e}")
            continue
        
        if len(content) < min_chunk_size:
            logger.warning(f"Skipping {file_path.name}: content too short ({len(content)} chars < {min_chunk_size})")
            continue
        
        # Generate num_q chunks from this file
        for _ in range(num_q):
            # Randomly select chunk position
            max_start = max(0, len(content) - min_chunk_size)
            if max_start == 0:
                start_pos = 0
            else:
                start_pos = random.randint(0, max_start)
            
            # Extract chunk using sentence-aware logic
            chunk_text = _extract_sentence_aware_chunk(
                content=content,
                start_pos=start_pos,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size
            )
            
            # Skip empty chunks
            if not chunk_text:
                logger.warning(f"Skipping empty chunk from {file_path.name}")
                continue
            
            chunks_to_process.append({
                "text": chunk_text,
                "document_name": file_path.name,
                "file_path": file_path_str
            })
    
    if not chunks_to_process:
        raise ValidationError("No valid chunks extracted from documents")
    
    logger.info(f"Processing {len(chunks_to_process)} chunks...")
    
    # Generate questions in batches
    results = []
    passages = [chunk["text"] for chunk in chunks_to_process]
    
    # Process in batches with progress bar
    qa_pairs = qa_generator.generate_qa_batch(passages, max_retries=1)
    
    # Combine with document metadata
    for i, (chunk_info, qa_pair) in enumerate(zip(chunks_to_process, qa_pairs)):
        if qa_pair and "question" in qa_pair:
            results.append({
                "document_name": chunk_info["document_name"],
                "file_path": chunk_info["file_path"],
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "chunk": chunk_info["text"]
            })
        else:
            logger.warning(f"Failed to generate question for chunk from {chunk_info['document_name']}")
    
    if not results:
        raise ValidationError("No questions were successfully generated")
    
    # Warn if we generated fewer questions than requested (only for num_questions mode)
    if num_questions != -1 and questions_per_doc is None and len(results) < num_questions:
        logger.warning(
            f"Generated {len(results)} questions, which is less than requested {num_questions}. "
            f"This may be due to short documents or failed question generation."
        )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate CSV filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_filename = output_path / f"questions_{timestamp}.jsonl"
    
    # Write CSV file using pandas
    try:
        df = pd.DataFrame(results)
        # Use QUOTE_ALL (quoting=1) which quotes all fields
        # This is the most robust way to handle special characters without needing escapechar
        # pandas will automatically handle quote escaping by doubling them
        df.to_json(jsonl_filename, orient="records", lines=True, force_ascii=False)

    except Exception as e:
        raise ValidationError(f"Failed to write CSV file: {e}") from e
    
    logger.info(f"Generated {len(results)} questions and saved to {jsonl_filename}")
    
    return str(jsonl_filename)

