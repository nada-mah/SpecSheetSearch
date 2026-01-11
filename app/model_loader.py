import os
import logging
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
def get_yolo_model_path():
    """
    Returns the path to the YOLO model, downloading it if necessary.
    
    Uses local path if exists, otherwise downloads from Hugging Face Hub.
    """
    model_path = "models/models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"

    logger.debug(f"Checking for existing model at: {model_path}")

    if not os.path.exists(model_path):
        logger.info("Model (doclayout_yolo) not found locally. Downloading from Hugging Face Hub...")
        try:
            model_path = hf_hub_download(
                repo_id="opendatalab/PDF-Extract-Kit-1.0",
                filename="models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt",
                local_dir="./models"
            )
            logger.info(f"Model downloaded and saved at: {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model from Hugging Face Hub: {e}", exc_info=True)
            raise

    else:
        logger.debug(f"Using existing model at: {model_path}")

    return model_path

# qwen_model_instance = None
# tokenizer_instance = None
    # global qwen_model_instance, tokenizer_instance
    # if qwen_model_instance is None or tokenizer_instance is None:
    #     logger.info("Model (Qwen3-8B-unsloth-bnb-4bit) not found locally. Downloading...")
    #     qwen_model_instance, tokenizer_instance = FastLanguageModel.from_pretrained(
    #         model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
    #         max_seq_length=16384,
    #         load_in_4bit=True,
    #         load_in_8bit=False,
    #         full_finetuning=False,
    #     )
    # return qwen_model_instance, tokenizer_instance
def get_qwen_model_path(use_gpu=False):
    """
    Returns the path to the Qwen model. Uses GPU-specific repo/filename if use_gpu=True.
    """
    logger.info("loading qwen...")
    
    # Select config based on GPU flag
    repo_id = LLM_REPO_ID_GPU if use_gpu else LLM_REPO_ID
    filename = LLM_FILENAME_GPU if use_gpu else LLM_FILENAME
    local_dir = "./models/qwen_gpu" if use_gpu else "./models/qwen"

    logger.info(f"Loading {'GPU' if use_gpu else 'CPU'} model: {repo_id}/{filename}")


    # Download the file
    local_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir
    )

    logger.info(f"Model downloaded to: {local_file_path}")
    return local_file_path

def get_llm_instance(use_gpu=False):
    global llm
    if use_gpu:
        n_gpu = -1
    else:
        n_gpu = 0
    if llm is None:
        llm_path = get_qwen_model_path(use_gpu)
        llm = Llama(
        model_path=llm_path,
        n_gpu_layers=n_gpu,         # Offload all possible layers to the GPU
        n_batch=512,             # Process up to 512 tokens in parallel
        n_ctx=8192,             # Context window size
        verbose=False            # Set to True to see detailed loading information
        )
    return llm

def get_ocr_instance():
    """
    Returns the global OCR instance, initializing it if necessary.
    """
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("Initializing PaddleOCR model...")
        _ocr_instance = PaddleOCR(
            lang="en",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # text_det_limit_side_len=640,
            text_recognition_batch_size=16,
        )
    return _ocr_instance