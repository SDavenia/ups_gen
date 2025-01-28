# Additional utils.
import torch
import logging
import random
import json
import pathlib
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


def reorder_column(df_filtered, column, reference_list, add_id=False):
    """
    Reorders the values in a specified column of a DataFrame according to the order defined in a reference list.

    Parameters:
        df_filtered (pd.DataFrame): The DataFrame containing the column to reorder.
        column (str): The name of the column in the DataFrame to reorder.
        reference_list (list): A list specifying the desired order of values in the column.
        add_id (bool, optional): If True, adds a column ('sort_order') with the numeric mapping based on the reference list.
                                Default is False.

    Returns:
        pd.DataFrame: A reordered DataFrame with rows rearranged based on the specified column order.

    Raises:
        ValueError: If any values in the specified column are not found in the reference list.
    """
    # Create a mapping from objects in the reference list to their position in the reference list.
    reference_order = {x.strip(): idx for idx, x in enumerate(reference_list)}
    id_column_name = f"{column}_id"
    
    # Create a column containing the numeric values which are used for sorting
    df_filtered[id_column_name] = df_filtered[column].map(reference_order)
    if add_id:
        df_filtered_sorted = df_filtered.sort_values(id_column_name)
    else:
        df_filtered_sorted = df_filtered.sort_values(id_column_name).drop('sort_order', axis=1)
    
    # Ensure all rows are matched to the reference list.
    if len(df_filtered_sorted) != len(df_filtered):
        raise ValueError("Some propositions couldn't be matched to the reference list")
    
    return df_filtered_sorted.reset_index(drop=True)


def count_values(df,
                 column_name):
    """
    Count occurrences of each unique value in the specified column_name column
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing specified column_name column
    
    Returns:
    pandas.Series: Count of each unique value
    """
    # Count values including NaN
    value_counts = df[column_name].value_counts(dropna=False)
    
    return value_counts


def fix_label(value, label_fixes) -> str:
    """
    Given a dict of label fixes, return the fixed label if it exists, otherwise return 'None'.
    Args:
        value (str): The label to fix.
        label_fixes (dict): A dict of label fixes.
    Returns:
        str: The fixed label.
    """
    value_tomatch = ''.join(e for e in value.lower().strip() if e.isalnum() or e.isspace())
    if value_tomatch in label_fixes.keys():
        return label_fixes[value_tomatch]
    return 'None'


def read_json(file_path: pathlib.Path):
    with open(file_path) as f:
        return json.load(f)
    
def read_lines(file_path: pathlib.Path):
    with open(file_path) as f:
        return f.readlines()
