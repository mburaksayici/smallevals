"""Model loader for llama-cpp-python inference without HuggingFace dependencies."""

import os
import platform
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from llama_cpp import Llama

from smallevals.exceptions import ModelLoadError
from smallevals.utils.logger import logger


# Hardcoded model configuration
HF_REPO_ID = "mburaksayici/golden_generate_qwen_0.5"
HF_FILENAME = "qwen3-0.6b.Q4_K_M.gguf"


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
    
    # Download the file with progress bar
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
    Detect and configure device for llama-cpp-python.
    
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
    # llama-cpp-python with n_gpu_layers=-1 will automatically use:
    # - CUDA on Linux/Windows with NVIDIA GPU
    # - Metal on Mac (MPS)
    # If GPU is not available, it will fall back to CPU automatically
    is_mac = platform.system() == "Darwin"
    
    # Try GPU/MPS first (n_gpu_layers=-1 works for both)
    # The actual fallback to CPU will happen in __init__ if loading fails
    if is_mac:
        return ("mps", -1)  # Mac: try Metal/MPS first
    else:
        return ("cuda", -1)  # Linux/Windows: try CUDA first


class ModelLoader:
    """Loads models using llama-cpp-python with direct file paths or URLs."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        n_ctx: int = 768,
        n_gpu_layers: Optional[int] = None,
    ):
        """
        Initialize model loader.

        Args:
            model_path: Optional path to GGUF model file or URL. If None, uses hardcoded HF repo.
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
                   Auto-detect: tries GPU first, then MPS (if Mac), then CPU
            batch_size: Batch size for inference (processed sequentially)
            n_ctx: Context length (default: 2048)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all layers, 0 for CPU only).
                         If None, determined from device parameter.
        """
        # Validate batch_size
        if batch_size <= 0:
            raise ModelLoadError(f"batch_size must be positive, got {batch_size}")
        if n_ctx <= 0:
            raise ModelLoadError(f"n_ctx must be positive, got {n_ctx}")
        
        # Use hardcoded model if path not provided
        if model_path is None:
            model_url = get_hf_model_url(HF_REPO_ID, HF_FILENAME)
            logger.info(f"Using hardcoded model: {HF_REPO_ID}/{HF_FILENAME}")
        else:
            model_url = model_path
        
        self.batch_size = batch_size
        self.n_ctx = n_ctx
        
        # Detect device and set n_gpu_layers
        if n_gpu_layers is None:
            device_name, n_gpu_layers = detect_device(device)
            self.device = device_name
            logger.info(f"Auto-detected device: {device_name}")
        else:
            self.device = device or "auto"
            logger.info(f"Using n_gpu_layers={n_gpu_layers} (device={self.device})")
        
        self.n_gpu_layers = n_gpu_layers

        # Download model if it's a URL
        if model_url.startswith("http://") or model_url.startswith("https://"):
            actual_model_path = download_model_file(model_url)
        elif not os.path.exists(model_url):
            raise ModelLoadError(f"Model file not found: {model_url}")
        else:
            actual_model_path = model_url

        # Load model with llama-cpp-python
        logger.info(f"Loading model from {actual_model_path}...")
        try:
            self.llm = Llama(
                model_path=actual_model_path,
                n_gpu_layers=n_gpu_layers,  # -1 for GPU/MPS, 0 for CPU
                n_ctx=n_ctx,  # Context length
                verbose=False
            )
            logger.info(f"Model loaded successfully (device={self.device}, n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx})")
        except Exception as e:
            # If GPU fails, try falling back to CPU
            if n_gpu_layers != 0:
                logger.warning(f"Failed to load with GPU (n_gpu_layers={n_gpu_layers}), falling back to CPU...")
                self.n_gpu_layers = 0
                self.device = "cpu"
                try:
                    self.llm = Llama(
                        model_path=actual_model_path,
                        n_gpu_layers=0,
                        n_ctx=n_ctx,
                        verbose=False
                    )
                    logger.info(f"Model loaded successfully on CPU (n_ctx={n_ctx})")
                except Exception as cpu_error:
                    raise ModelLoadError(f"Failed to load model on CPU: {cpu_error}") from cpu_error
            else:
                raise ModelLoadError(f"Failed to load model: {e}") from e

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 400,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = ["<|im_end|>", "User:", "</s>"],
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            stop_sequences: Optional list of stop sequences

        Returns:
            List of generated responses
        """
        if not prompts:
            return []

        results = []
        for prompt in prompts:
            # Generate response
            output = self.llm(
                prompt,
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

        return results

    def generate_batched(
        self,
        prompts: List[str],
        max_new_tokens: int = 400,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = ["<|im_end|>", "User:", "</s>"],
    ) -> List[str]:
        """
        Generate responses for prompts in batches.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional list of stop sequences

        Returns:
            List of generated responses
        """
        if not prompts:
            return []
        
        all_results = []
        num_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(prompts), desc="Generating", unit="prompt") as pbar:
            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i : i + self.batch_size]
                batch_results = self.generate(
                    batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                )
                all_results.extend(batch_results)
                pbar.update(len(batch))

        return all_results

