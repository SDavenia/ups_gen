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
    # For calling the logger from generate_answers.py
    if hasattr(args, 'model_id'):
        model_id = args.model_id
    elif hasattr(args, 'model_data_id'):
        model_id = args.model_data_id
    else:
        raise ValueError("The arguments should contain either model_id or model_data_id.")
    logger = logging.getLogger(f"model_{model_id.replace('/', '_')}")  # Replace '/' with '_' for a valid logger name
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
    """
    If the model tokenizer does not contain a pad token, use the eos token as the pad token.
    
    Args:
        model_id (str): The model id.
        model (transformers.PreTrainedModel): The model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
    """
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        logging.info(f"Using tokenizer.eos_token as the pad token for {model_id}.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    else:
        raise NotImplementedError(f"The tokenizer for {model_id} does not have a pad token.")
    return model, tokenizer

def load_model(model_id,
               device,
               precision=torch.float16,
               quantization_config=None):
    """
    Load the model and tokenizer from the model id.

    Args:
        model_id (str): The model id.
        device (torch.device): The device to use.
        quantization_config (dict): The quantization configuration.
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=precision).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    
    if quantization_config is not None:
        raise NotImplementedError("Quantization is not supported yet.")

    if tokenizer.pad_token is not None and tokenizer.pad_token_id is not None:
        return model, tokenizer
    else:
        model, tokenizer = add_pad_token(model_id, model, tokenizer)
        return model, tokenizer
