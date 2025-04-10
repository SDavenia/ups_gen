import torch
from typing import List
from tqdm import tqdm


def run_prompts(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    batch_size: int = 1,
    **generation_kwargs,
) -> List[str]:  # Return type is a list of generated responses
    """
    Takes a model, a tokenizer, a list of prompts, and a device and generates responses for each prompt using the specified batch size.
    The sentences should be formatted correctly for the specific model.

    Args:
        model: A Hugging Face model to generate responses.
        tokenizer: A Hugging Face tokenizer to encode the prompts.
        prompts (list): A list of prompts to generate responses for.
        device (torch.device): The device (CPU or GPU) to run the model on.
        batch_size (int, optional): The batch size to use for generation. Default is 1.
        **generation_kwargs: Additional generation arguments (e.g., max_length, num_beams, temperature) to pass to the model.

    Returns:
        list: A list of generated responses corresponding to the input prompts.
    """
    all_outputs = []

    for i in tqdm(
        range(0, len(prompts), batch_size), desc="Processing batches", unit="batch"
    ):
        batch_prompts = prompts[i : i + batch_size]
        
        # Prepare the inputs by tokenizing the batch of prompts
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        # Generate outputs using the model
        # Llama-3-Instruct family of models require special handling for terminators.
        if model.config._name_or_path == "meta-llama/Llama-3.1-8B-Instruct" or model.config._name_or_path == "meta-llama/Llama-3.1-70B-Instruct" or model.config._name_or_path == "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated":
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = model.generate(**inputs, **generation_kwargs, eos_token_id=terminators)
        else:
            outputs = model.generate(**inputs, **generation_kwargs)
        # Retain only the actual generation
        outputs_generation = outputs[:, inputs.input_ids.shape[1]:]
        
        # Decode the generated tokens back into text and skip special tokens.
        decoded_outputs_generation = tokenizer.batch_decode(outputs_generation, skip_special_tokens=True)
        
        # Extend the all_outputs list with the newly decoded responses
        all_outputs.extend(decoded_outputs_generation)
    
    return all_outputs
