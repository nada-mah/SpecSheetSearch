import os
import logging
import sys
import pathlib
from ctypes import CFUNCTYPE, c_int, c_char_p, c_void_p
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from paddleocr import PaddleOCR
from config import (
    LLM_FILENAME, LLM_REPO_ID,
    LLM_FILENAME_GPU, LLM_REPO_ID_GPU  # Import GPU versions
)
# from unsloth import FastLanguageModel
# from config.logging_config import  setup_logger

logger = logging.getLogger(__name__)

# Global variable to hold the OCR instance
_ocr_instance = None
llm = None

# ============================================================================
# llama.cpp Log Callback Setup (Captures internal C logs to Python logger)
# ============================================================================

try:
    from llama_cpp.llama_cpp import llama_log_set, GGML_LOG_LEVEL_ERROR, GGML_LOG_LEVEL_WARN, GGML_LOG_LEVEL_INFO, GGML_LOG_LEVEL_DEBUG
    
    # Define the callback function type signature
    LOG_CALLBACK_TYPE = CFUNCTYPE(None, c_int, c_char_p, c_void_p)
    
    def _llama_log_callback(level: int, text: bytes, user_data: int):
        """Callback to redirect llama.cpp logs to Python logger"""
        try:
            message = text.decode('utf-8', errors='replace').strip()
            if not message:
                return
            
            # Map llama.cpp log levels to Python logging levels
            if level <= GGML_LOG_LEVEL_ERROR:
                logger.error(f"[llama.cpp] {message}")
            elif level <= GGML_LOG_LEVEL_WARN:
                logger.warning(f"[llama.cpp] {message}")
            elif level <= GGML_LOG_LEVEL_INFO:
                logger.info(f"[llama.cpp] {message}")
            else:
                logger.debug(f"[llama.cpp] {message}")
        except Exception as e:
            # Fallback to stderr if logging fails
            print(f"[llama.cpp log error] {e}: {text}", file=sys.stderr)
    
    # Keep a reference to prevent garbage collection
    _log_callback_ref = LOG_CALLBACK_TYPE(_llama_log_callback)
    
    # Register the callback (do this once at module load)
    llama_log_set(_log_callback_ref, None)
    logger.debug("✓ Registered llama.cpp log callback handler")
    
except ImportError as e:
    logger.warning(f"⚠ Could not import llama_cpp log functions: {e}")
    logger.warning("⚠ Internal llama.cpp logs will only appear if verbose=True")
except Exception as e:
    logger.warning(f"⚠ Failed to register llama.cpp log callback: {e}")


# ============================================================================
# GPU Support Detection
# ============================================================================

def check_gpu_support() -> bool:
    """
    Check if llama_cpp was compiled with GPU support (CUDA/Metal/ROCm).
    Returns True if GPU offloading is available.
    """
    try:
        from llama_cpp.llama_cpp import _load_shared_library
        
        # Try to load the llama shared library
        lib = _load_shared_library('llama')
        
        # Check for GPU offload support function
        if hasattr(lib, 'llama_supports_gpu_offload'):
            result = bool(lib.llama_supports_gpu_offload())
            logger.debug(f"GPU offload support check: {result}")
            return result
        else:
            logger.debug("llama_supports_gpu_offload function not found in library")
            return False
            
    except ImportError:
        logger.debug("Could not import _load_shared_library - trying fallback")
    except Exception as e:
        logger.debug(f"GPU support check failed (expected on some builds): {e}")
    
    # Fallback: check environment/CMAKE args hint
    cmake_args = os.environ.get('CMAKE_ARGS', '')
    if 'GGML_CUDA=on' in cmake_args or 'GGML_METAL=on' in cmake_args or 'GGML_ROCM=on' in cmake_args:
        logger.info("GPU support likely enabled via CMAKE_ARGS")
        return True
    
    logger.warning("Could not definitively determine GPU support - verify llama_cpp build")
    return False


# ============================================================================
# Model Loading Functions
# ============================================================================

def get_yolo_model_path():
    """
    Returns the path to the YOLO model, downloading it if necessary.
    
    Uses local path if exists, otherwise downloads from Hugging Face Hub.
    """
    model_path = "models/models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"

    logger.debug(f"Checking for existing YOLO model at: {model_path}")

    if not os.path.exists(model_path):
        logger.info("Model (doclayout_yolo) not found locally. Downloading from Hugging Face Hub...")
        try:
            model_path = hf_hub_download(
                repo_id="opendatalab/PDF-Extract-Kit-1.0",
                filename="models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt",
                local_dir="./models"
            )
            logger.info(f"✓ YOLO model downloaded and saved at: {model_path}")
        except Exception as e:
            logger.error(f"✗ Failed to download YOLO model from Hugging Face Hub: {e}", exc_info=True)
            raise
    else:
        logger.debug(f"✓ Using existing YOLO model at: {model_path}")

    return model_path


def get_qwen_model_path(use_gpu=False):
    """
    Returns the path to the Qwen model. Uses GPU-specific repo/filename if use_gpu=True.
    """
    logger.info("🔄 Loading Qwen model configuration...")
    
    # Select config based on GPU flag
    repo_id = LLM_REPO_ID_GPU if use_gpu else LLM_REPO_ID
    filename = LLM_FILENAME_GPU if use_gpu else LLM_FILENAME
    local_dir = "./models/qwen_gpu" if use_gpu else "./models/qwen"

    logger.info(f"  Mode: {'🟢 GPU' if use_gpu else '🔵 CPU'}")
    logger.info(f"  Repo ID: {repo_id}")
    logger.info(f"  Filename: {filename}")
    logger.info(f"  Local directory: {local_dir}")

    # Download the file
    local_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir
    )

    logger.info(f"✓ Model ready at: {local_file_path}")
    return local_file_path


def get_llm_instance(use_gpu=False):
    """
    Returns the global LLM instance, initializing it with GPU/CPU configuration.
    
    Args:
        use_gpu: If True, attempts to offload layers to GPU
        
    Returns:
        Llama: Initialized llama_cpp Llama instance
    """
    global llm
    
    if llm is not None:
        logger.debug("✓ Returning cached LLM instance")
        return llm

    logger.info("🚀 Initializing LLM instance...")
    
    # Determine GPU layer offloading setting
    n_gpu_layers = -1 if use_gpu else 0
    mode_str = "🟢 GPU" if use_gpu else "🔵 CPU"
    
    # Check actual GPU support before loading
    gpu_available = check_gpu_support()
    
    if use_gpu and not gpu_available:
        logger.warning(
            f"⚠ {mode_str} requested but llama_cpp may not be compiled with GPU support.\n"
            f"💡 To enable CUDA: pip install llama-cpp-python --no-cache-dir -C cmake.args='-DGGML_CUDA=on'\n"
            f"💡 To enable Metal (Mac): pip install llama-cpp-python --no-cache-dir -C cmake.args='-DGGML_METAL=on'"
        )
    
    # Get model path
    llm_path = get_qwen_model_path(use_gpu)
    
    # Log initialization parameters
    logger.info(f"📋 Llama initialization parameters:")
    logger.info(f"  • model_path: {llm_path}")
    logger.info(f"  • n_gpu_layers: {n_gpu_layers} ({'all layers' if n_gpu_layers == -1 else 'CPU only' if n_gpu_layers == 0 else f'{n_gpu_layers} layers'})")
    logger.info(f"  • n_batch: 512")
    logger.info(f"  • n_ctx: 12288")
    logger.info(f"  • verbose: True (llama.cpp internal logs enabled)")
    logger.info(f"  • GPU backend available: {gpu_available}")
    
    # Initialize the model with verbose=True to capture internal logs via callback
    try:
        llm = Llama(
            model_path=llm_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=512,
            n_ctx=12288,
            verbose=True,  # Enables internal llama.cpp logging (captured by our callback)
        )
        logger.info("✓ LLM model loaded successfully")
        
        # Post-load verification hints
        if use_gpu and gpu_available:
            logger.info("💡 GPU mode active - monitor with: watch -n 1 nvidia-smi")
        elif use_gpu and not gpu_available:
            logger.warning("⚠ Model loaded on CPU despite GPU request - check llama_cpp build")
        else:
            logger.info("✓ CPU mode active")
            
    except Exception as e:
        logger.error(f"✗ Failed to initialize LLM: {e}", exc_info=True)
        # Clear the global variable to allow retry
        llm = None
        raise
    
    return llm


def get_ocr_instance():
    """
    Returns the global OCR instance, initializing it if necessary.
    """
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("🔄 Initializing PaddleOCR model...")
        try:
            _ocr_instance = PaddleOCR(
                lang="en",
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_recognition_batch_size=16,
            )
            logger.info("✓ PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize PaddleOCR: {e}", exc_info=True)
            raise
    else:
        logger.debug("✓ Returning cached OCR instance")
    return _ocr_instance