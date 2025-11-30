"""Golden Generator for Qwen3 model inference with GGUF and HuggingFace transformers backends."""

import os
import platform
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from tqdm import tqdm

from smallevals.exceptions import ModelLoadError
from smallevals.utils.logger import logger


# Model configuration
HF_REPO_ID_GGUF = "mburaksayici/golden_generate_qwen_0.6b_v2_gguf"
HF_REPO_ID_HF = "mburaksayici/golden_generate_qwen_0.6b_v2"


class ModelVariant(Enum):
    """Model variant options for GGUF files."""
    Q4KM = "Qwen3-0.6B.Q4_K_M.gguf"
    F16 = "Qwen3-0.6B.F16.gguf"
    Q8_0 = "Qwen3-0.6B.Q8_0.gguf"


def get_hf_model_url(repo_id: str, filename: str) -> str:
    """
    Construct HuggingFace direct download URL without using hf_hub_download.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        filename: Name of the file to download
    
    Returns:
        Direct download URL for the file
    """
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def download_model_file(url: str, local_path: Optional[str] = None) -> str:
    """
    Download a model file from a URL to a local path.
    
    Args:
        url: URL to download the model file from
        local_path: Optional local path to save the file. If None, uses filename from URL.
    
    Returns:
        Path to the downloaded model file
    """
    if local_path is None:
        # Extract filename from URL
        filename = url.split("/")[-1]
        # Use a models directory in the user's home or current directory
        models_dir = Path.home() / ".smallevals" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        local_path = str(models_dir / filename)
    
    local_path = Path(local_path)
    
    # Skip download if file already exists
    if local_path.exists():
        logger.info(f"Model file already exists at {local_path}, skipping download.")
        return str(local_path)
    
    logger.info(f"Downloading model from {url} to {local_path}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading model") as pbar:
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                    pbar.update(block_size)
            
            urllib.request.urlretrieve(url, str(local_path), show_progress)
        
        logger.info(f"Model downloaded to {local_path}")
    except Exception as e:
        # Clean up partial download
        if local_path.exists():
            local_path.unlink()
        raise ModelLoadError(f"Failed to download model from {url}: {e}") from e
    
    return str(local_path)


def detect_device(device: Optional[str] = None) -> tuple[str, int]:
    """
    Detect and configure device for model inference.
    
    Args:
        device: Optional device string ("cuda", "mps", "cpu", or None for auto-detect)
    
    Returns:
        Tuple of (device_name, n_gpu_layers)
        - device_name: "cuda", "mps", or "cpu"
        - n_gpu_layers: -1 for GPU/MPS, 0 for CPU
    """
    if device is not None:
        device_lower = device.lower()
        if device_lower == "cuda":
            return ("cuda", -1)
        elif device_lower == "mps":
            return ("mps", -1)
        elif device_lower == "cpu":
            return ("cpu", 0)
        else:
            logger.warning(f"Unknown device '{device}', falling back to auto-detect")
    
    # Auto-detect: Try GPU first (works for both CUDA and Metal/MPS on Mac)
    is_mac = platform.system() == "Darwin"
    
    if is_mac:
        return ("mps", -1)  # Mac: try Metal/MPS first
    else:
        return ("cuda", -1)  # Linux/Windows: try CUDA first


def detect_backend(backend: Optional[str] = None) -> Literal["gguf", "transformers"]:
    """
    Detect available backend or use specified one.
    
    Args:
        backend: Optional backend string ("gguf", "transformers", or None for auto-detect)
    
    Returns:
        Backend name: "gguf" or "transformers"
    
    Raises:
        ModelLoadError: If neither backend is available
    """
    if backend is not None:
        backend_lower = backend.lower()
        if backend_lower in ("gguf", "llama-cpp"):
            try:
                import llama_cpp
                return "gguf"
            except ImportError:
                raise ModelLoadError(
                    "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
                )
        elif backend_lower == "transformers":
            try:
                import transformers
                return "transformers"
            except ImportError:
                raise ModelLoadError(
                    "transformers is not installed. Install it with: pip install transformers"
                )
        else:
            raise ModelLoadError(f"Unknown backend '{backend}'. Use 'gguf' or 'transformers'")
    
    # Auto-detect: try GGUF first, then transformers
    try:
        import llama_cpp
        logger.info("Auto-detected backend: GGUF (llama-cpp-python)")
        return "gguf"
    except ImportError:
        logger.info("llama-cpp-python not available, trying transformers...")
    
    try:
        import transformers
        logger.info("Auto-detected backend: transformers")
        return "transformers"
    except ImportError:
        raise ModelLoadError(
            "Neither llama-cpp-python nor transformers is available. "
            "Install at least one: pip install llama-cpp-python OR pip install transformers"
        )


def format_qwen3_prompt(prompt: str, system_message: Optional[str] = None) -> str:
    """
    Format prompt with Qwen3 chat template.
    
    Args:
        prompt: User prompt
        system_message: Optional system message
    
    Returns:
        Formatted prompt string
    """
    if system_message:
        return f"""<|im_start|>system
{system_message}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
    else:
        return f"""<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""


class GoldenGenerator:
    """Golden Generator for Qwen3 model inference with GGUF and HuggingFace transformers backends."""

    def __init__(
        self,
        model_variant: ModelVariant = ModelVariant.Q4KM,
        backend: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        n_ctx: int = 32768,
        n_gpu_layers: Optional[int] = None,
    ):
        """
        Initialize Golden Generator.

        Args:
            model_variant: Model variant to use (default: Q4KM)
            backend: Backend to use ("gguf", "transformers", or None for auto-detect)
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
                   Auto-detect: tries GPU first, then MPS (if Mac), then CPU
            batch_size: Batch size for inference (processed sequentially)
            n_ctx: Context length (default: 768 for GGUF, 2048 for transformers)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all layers, 0 for CPU only).
                         If None, determined from device parameter. Only used for GGUF backend.
        """
        if batch_size <= 0:
            raise ModelLoadError(f"batch_size must be positive, got {batch_size}")
        if n_ctx <= 0:
            raise ModelLoadError(f"n_ctx must be positive, got {n_ctx}")
        
        self.model_variant = model_variant
        self.batch_size = batch_size
        self.n_ctx = n_ctx
        
        self.backend = detect_backend(backend)
        logger.info(f"Using backend: {self.backend}")
        
        if n_gpu_layers is None:
            device_name, n_gpu_layers = detect_device(device)
            self.device = device_name
            logger.info(f"Auto-detected device: {device_name}")
        else:
            self.device = device or "auto"
            logger.info(f"Using n_gpu_layers={n_gpu_layers} (device={self.device})")
        
        self.n_gpu_layers = n_gpu_layers
        
        if self.backend == "gguf":
            self._load_gguf_model()
        else:
            self._load_transformers_model()
    
    def _load_gguf_model(self):
        """Load model using llama-cpp-python (GGUF backend)."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ModelLoadError("llama-cpp-python is not installed")
        
        filename = self.model_variant.value
        model_url = get_hf_model_url(HF_REPO_ID_GGUF, filename)
        logger.info(f"Using GGUF model: {HF_REPO_ID_GGUF}/{filename}")
        
        if model_url.startswith("http://") or model_url.startswith("https://"):
            actual_model_path = download_model_file(model_url)
        elif not os.path.exists(model_url):
            raise ModelLoadError(f"Model file not found: {model_url}")
        else:
            actual_model_path = model_url

        logger.info(f"Loading GGUF model from {actual_model_path}...")
        try:
            self.llm = Llama(
                model_path=actual_model_path,
                n_gpu_layers=self.n_gpu_layers,  # -1 for GPU/MPS, 0 for CPU
                n_ctx=self.n_ctx,  # Context length
                verbose=False
            )
            logger.info(f"GGUF model loaded successfully (device={self.device}, n_gpu_layers={self.n_gpu_layers}, n_ctx={self.n_ctx})")
        except Exception as e:
            if self.n_gpu_layers != 0:
                logger.warning(f"Failed to load with GPU (n_gpu_layers={self.n_gpu_layers}), falling back to CPU...")
                self.n_gpu_layers = 0
                self.device = "cpu"
                try:
                    self.llm = Llama(
                        model_path=actual_model_path,
                        n_gpu_layers=0,
                        n_ctx=self.n_ctx,
                        verbose=False
                    )
                    logger.info(f"GGUF model loaded successfully on CPU (n_ctx={self.n_ctx})")
                except Exception as cpu_error:
                    raise ModelLoadError(f"Failed to load GGUF model on CPU: {cpu_error}") from cpu_error
            else:
                raise ModelLoadError(f"Failed to load GGUF model: {e}") from e
    
    def _load_transformers_model(self):
        """Load model using HuggingFace transformers backend."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ModelLoadError(f"transformers or torch is not installed: {e}")
        
        logger.info(f"Using HuggingFace transformers model: {HF_REPO_ID_HF}")
        
        if self.device == "cuda":
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
        elif self.device == "mps":
            device_map = "mps" if torch.backends.mps.is_available() else "cpu"
            if not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available, falling back to CPU")
        else:
            device_map = "cpu"
        
        try:
            logger.info(f"Loading transformers model (device={device_map})...")
            self.tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID_HF)
            self.model = AutoModelForCausalLM.from_pretrained(
                HF_REPO_ID_HF,
                device_map=device_map,
                torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
            )
            logger.info(f"Transformers model loaded successfully (device={device_map})")
        except Exception as e:
            raise ModelLoadError(f"Failed to load transformers model: {e}") from e

    def _truncate_prompt_if_needed(
        self, 
        formatted_prompt: str, 
        max_characters: int = 7000
    ) -> str:
        """
        Truncate prompt from the end if prompt exceeds max_characters.
        
        Args:
            formatted_prompt: The formatted prompt string
            max_characters: Maximum total context characters (default: 7000)
            
        Returns:
            Truncated prompt with '\nReturn ONLY a JSON object.' appended if truncated, or original prompt
        """
        # Reserve space for the suffix we'll add after truncation
        suffix = '\nReturn ONLY a JSON object.'
        suffix_length = len(suffix)
        
        prompt_chars = len(formatted_prompt)
        
        # Calculate max prompt chars (reserving space for suffix)
        max_prompt_chars = max_characters - suffix_length
        
        if max_prompt_chars <= 0:
            logger.warning(
                f"max_characters ({max_characters}) is too small to fit the suffix. "
                f"Using minimum size."
            )
            max_prompt_chars = max(1, max_characters // 2)
        
        # If prompt fits within the limit, return as-is
        if prompt_chars <= max_prompt_chars:
            return formatted_prompt
        
        # Truncate from the end
        truncated_prompt = formatted_prompt[:max_prompt_chars]
        
        # Append the suffix
        truncated_prompt = truncated_prompt + suffix
        
        final_chars = len(truncated_prompt)
        logger.warning(
            f"Truncated prompt from {prompt_chars} to {final_chars} characters "
            f"(removed {prompt_chars - max_prompt_chars} characters from end to fit within {max_characters} context)"
        )
        
        return truncated_prompt

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 7000,
        temperature: float = 0.8,
        stop_sequences: Optional[List[str]] = ["<|im_end|>"],
        system_message: Optional[str] = None,
        max_chars_in_prompts: int = 6000,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            stop_sequences: Optional list of stop sequences
            system_message: Optional system message to prepend to prompts
            max_chars_in_prompts: Maximum total context tokens. If prompt + max_new_tokens exceeds
                               this, the prompt will be truncated from the end (default: 6000)

        Returns:
            List of generated responses
        """
        if not prompts:
            return []

        results = []
        for prompt in prompts:
            # Format prompt with Qwen3 chat template
            formatted_prompt = format_qwen3_prompt(prompt, system_message)
            
            # Truncate prompt if it exceeds context window
            formatted_prompt = self._truncate_prompt_if_needed(
                formatted_prompt, 
                max_characters=max_chars_in_prompts
            )
            
            if self.backend == "gguf":
                # Generate response using GGUF backend
                output = self.llm(
                    formatted_prompt,
                    max_tokens=max_new_tokens,
                    stop=stop_sequences if stop_sequences else [],
                    temperature=temperature if temperature > 0 else 0.0,
                    echo=False,  # Don't echo the prompt in the output
                )
                
                # Extract generated text from response
                if output and "choices" in output and len(output["choices"]) > 0:
                    generated_text = output["choices"][0].get("text", "")
                    results.append(generated_text)
                else:
                    results.append("")
            else:
                # Generate response using transformers backend
                import torch
                
                # Tokenize input
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.n_ctx
                )
                
                # Move to device
                if hasattr(self.model, 'device'):
                    device = next(self.model.parameters()).device
                else:
                    device = torch.device("cpu")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else 1.0,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode output (skip the input tokens)
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Apply stop sequences if provided
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]
                
                results.append(generated_text.strip())

        return results

    def generate_batched(
        self,
        prompts: List[str],
        max_new_tokens: int = 400,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = ["<|im_end|>"],
        system_message: Optional[str] = None,
        max_chars_in_prompts: int = 6000,
    ) -> List[str]:
        """
        Generate responses for prompts in batches.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional list of stop sequences
            system_message: Optional system message to prepend to prompts
            max_chars_in_prompts: Maximum total context tokens. If prompt + max_new_tokens exceeds
                               this, the prompt will be truncated from the end (default: 6000)

        Returns:
            List of generated responses
        """
        if not prompts:
            return []
        
        all_results = []
        with tqdm(total=len(prompts), desc="Generating", unit="prompt") as pbar:
            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i : i + self.batch_size]
                batch_results = self.generate(
                    batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                    system_message=system_message,
                    max_chars_in_prompts=max_chars_in_prompts,
                )
                all_results.extend(batch_results)
                pbar.update(len(batch))

        return all_results

