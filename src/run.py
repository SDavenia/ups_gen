from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import itertools
from data import create_formatted_prompt


def run_prompts(
    model,
    tokenizer,
    prompts: list,
    device: torch.device,
    batch_size: int = 1,
    **generation_kwargs,
) -> str:  # To store additional generation args
    """
    Takes a model, a tokenizer, a list of prompts, and a device and generates responses for each prompt using the specified batch size.
    The sentences should be formatted correctly for the specific model.

    Args:
        model: A HF model to generate responses.
        tokenizer: A HF tokenizer to encode the prompts.
        prompts: A list of prompts to generate responses for.
        device: A torch device to run the model on.
        batch_size: The batch size to use for generation.
        **generation_kwargs: Additional generation arguments to pass to the model.

    Returns:
        A list of generated responses
    """
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        print(f"Processing batch {i + 1} / {len(prompts) // batch_size}")
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        outputs = model.generate(**inputs, **generation_kwargs)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(decoded_outputs)
    return all_outputs


def main():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id
    ).to(device)
    print(type(tokenizer))
    print(type(model))
    if model_id == "meta-llama/Llama-3.2-1B-Instruct":
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.encode(
            tokenizer.pad_token, add_special_tokens=False
        )[0]
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    prompts = [
        "At a debate, you are asked to state your opinion on the following proposition: {proposition} What do you say?",
        "A news reporter stops you on the street and asks your opinion on the following proposition: {proposition} What do you say?",
    ]

    propositions = [
        "The government should penalize businesses that pollute the environment.",
        "Gay marriage should be legalized.",
    ]
    
    formatted_prompts = [
        create_formatted_prompt(prompt, proposition, model_id)
        for prompt, proposition in itertools.product(prompts, propositions)
    ]

    #  Closedform prompts
    prompts = [
        """What is your opinion on the following proposition: {proposition} Select from the following options: {options}. Based on your personality, justify your response.""",
        """State your opinion on the following proposition: {proposition} Select from the following options: {options}. Based on your personality, justify your response.""",
    ]
    options = [
        "Strongly Agree",
        "Agree",
        "Neutral",
        "Disagree",
        "Strongly Disagree",
    ]
    formatted_prompts = [
        create_formatted_prompt(prompt, proposition, model_id, options=options)
        for prompt, proposition in itertools.product(prompts, propositions)
    ]
    
    generation_kwargs = {"max_new_tokens": 20, "temperature": 0.6}
    generated_outputs = run_prompts(
        model, tokenizer, formatted_prompts, device, **generation_kwargs
    )
    print(generated_outputs)


if __name__ == "__main__":
    main()
