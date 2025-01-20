# Additional utils.
import torch
import logging
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def prepare_logger(args) -> None:
    """
    Prepares a logger with id provided by the model name.

    Args:
        args: Command line arguments. It should contain the model_id.
    """
    logger = logging.getLogger(f"model_{args.model_id.replace('/', '_')}")  # Replace '/' with '_' for a valid logger name
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_reproducibility(seed: int) -> None:
    """
    Ensure reproducibility by setting random seeds for Python, Numpy, and PyTorch.

    Args:
        seed (int): The seed to use for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_pad_token(model_id, model, tokenizer):
    if model_id == "meta-llama/Llama-3.2-1B-Instruct":
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.encode(
            tokenizer.pad_token, add_special_tokens=False
        )[0]
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    else:
        raise NotImplementedError(f"The tokenizer for {model_id} does not have a pad token.")
    return model, tokenizer


def load_model(model_id, device, quantization_config=None):
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is not None and tokenizer.pad_token_id is not None:
        return model, tokenizer
    else:
        model, tokenizer = add_pad_token(model_id, model, tokenizer)
        return model, tokenizer
