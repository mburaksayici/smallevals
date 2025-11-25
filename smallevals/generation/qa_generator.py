"""QA generation from chunks using exact prompt format."""

from typing import List, Dict, Any, Optional
from tqdm import tqdm
from smallevals.models.loader import ModelLoader
from smallevals.utils.json_parser import parse_json_response
from smallevals.exceptions import ValidationError, QAGenerationError
from smallevals.utils.logger import logger


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
        
        # Uses hardcoded model from HuggingFace (configured in ModelLoader)
        self.model_loader = ModelLoader(device=device, batch_size=batch_size)

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
            prompts, max_new_tokens=400, temperature=0.0
        )

        # Parse responses
        results = []
        failed_indices = []
        for i, (passage, response) in enumerate(zip(passages, responses)):
            parsed = parse_json_response(response)
            
            if parsed and "question" in parsed and "answer" in parsed:
                results.append({
                    "question": parsed["question"],
                    "answer": parsed["answer"],
                    "passage": passage,
                })
            else:
                failed_indices.append(i)
                # Retry if parsing failed
                if max_retries > 0:
                    logger.debug(f"Retrying generation for passage {i}")
                    retry_result = self.generate_qa(passage)
                    if retry_result:
                        results.append(retry_result)
                    else:
                        logger.warning(f"Failed to generate QA for passage {i} after retry")
                        continue
                else:
                    logger.warning(f"Failed to generate QA for passage {i}")
                    continue

        if failed_indices:
            logger.warning(f"Failed to generate QA for {len(failed_indices)} passages: {failed_indices}")

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
        
        # Add metadata and chunk_id if available
        for i, (chunk, metadata) in enumerate(zip(chunks, chunk_metadata)):
            if i < len(qa_pairs):
                if metadata:
                    qa_pairs[i]["metadata"] = metadata
                # Preserve chunk_id if present
                if isinstance(chunk, dict):
                    if "id" in chunk:
                        qa_pairs[i]["chunk_id"] = chunk["id"]
                    elif "metadata" in chunk and isinstance(chunk["metadata"], dict) and "chunk_id" in chunk["metadata"]:
                        qa_pairs[i]["chunk_id"] = chunk["metadata"]["chunk_id"]

        return qa_pairs

