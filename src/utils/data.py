# Contains functions to prepare prompts to be given as inputs to the models.
import itertools
from typing import Tuple, List

SYSTEM_MESSAGE = "You are a helpful assistant."

FORMAT_STRING = {
    'closed_domain': "Format the output to JSON with the fields 'selection' with the option that you select and 'response' containing your response in unformatted plain text.",
    'open_domain': "Format all outputs to JSON with the field 'response' containing your response in unformatted plain text."
}

SEP_ADDITIONAL_CONTEXT = "\n\n"  # Separator between the additional context and the user/system part of the prompt.

PROMPT_TEMPLATES = {

    # Llama family of models
    "meta-llama/Llama-3.1-8B-Instruct": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    "meta-llama/Llama-3.2-1B-Instruct": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",

    # Mistral family of models
    "mistralai/Mistral-7B-Instruct-v0.3": """<s>[INST] {system_message}
{user_message} [/INST]""",

    "mistralai/Mixtral-8x7B-Instruct-v0.1": """<s>[INST] {system_message}
{user_message} [/INST]""",

    "HuggingFaceH4/zephyr-7b-beta": """<|system|>
{system_message}</s>
<|user|>
{user_message}</s>
<|assistant|>
""",

    # AllenAI models
    "allenai/OLMo-7B-Instruct": """<|system|>
{system_message}
<|user|>
{user_message}
<|assistant|>

""",

}


def create_formatted_prompts(
        prompts: list,
        propositions: list,
        model_id: str,
        format_to_json: bool,  # If True, the model is required to format output to json.
        options: list = None,  # Defaults to open generation where options are not needed
        additional_context: str = None,
        additional_context_placement: str = None,
) -> Tuple[List[str], List[Tuple]]:
    """
    Given a list of prompts, and proposition it combines them to return possible prompts combinations formatted into a prompt.
    It also returns a list of tuples (prompt, proposition) corresponding to each combination.
    """
    formatted_prompts = []
    prompt_proposition_list = []

    for prompt, proposition in itertools.product(prompts, propositions):
        formatted_prompt = create_formatted_prompt(
            prompt=prompt,
            proposition=proposition,
            model_id=model_id,
            format_to_json=format_to_json,
            options=options,
            additional_context=additional_context,
            additional_context_placement=additional_context_placement,
        )
        formatted_prompts.append(formatted_prompt)
        prompt_proposition_list.append((prompt, proposition))

    return formatted_prompts, prompt_proposition_list

def create_formatted_prompt(
    prompt: str,
    proposition: str,
    model_id: str,
    format_to_json: bool,
    options: list,
    additional_context: str,
    additional_context_placement: str,
) -> str:
    """
    Given a model, a prompt format, a proposition this function returns the corresponding formatted prompt template.

    Args:
        model_id (str): Needed for the prompt formatting with tags etc...
        prompt (str): The prompt to be used (from ../data/prompting/prompts_wright.json)
        proposition (str): The PCT question (from ../data/political_compass/political_compass_questions.txt)
        format_to_json (bool): If True, the model is required to format output to json.
        options (list): list of options if the model is prompted in the closed setting.
        additional_context (str, optional): Additional context to be added to the prompt to test the UPS
        additional_context_placement (str, optional): Where to place the additional context in the prompt.
                                                      Can be one of system, user

    Returns:
        str: The formatted prompt
    """

    prompt_template = PROMPT_TEMPLATES.get(model_id, None)
    if prompt_template is None:
        raise NotImplementedError(
            f"Prompt template for model_id '{model_id}' is not available in PROMPT_TEMPLATES."
        )
    # Prepare user input by passing the proposition and the options if in the closed setting.
    user_message = prompt.format(
        proposition=proposition, options=options
    )
    # Prepare the prompt to be given as input to the model including the system and user message.
    if format_to_json:
        system_message = f"{SYSTEM_MESSAGE}\n{FORMAT_STRING['closed_domain']}" if options is not None else f"{SYSTEM_MESSAGE}\n{FORMAT_STRING['open_domain']}"
    else:
        system_message = SYSTEM_MESSAGE
    
    # Add the additional context
    if additional_context_placement == "system-beginning":
        system_message = f"{additional_context}{SEP_ADDITIONAL_CONTEXT}{system_message}"
    elif additional_context_placement == "system-end":
        system_message = f"{system_message}{SEP_ADDITIONAL_CONTEXT}{additional_context}"
    elif additional_context_placement == "user-beginning":
        user_message = f"{additional_context}{SEP_ADDITIONAL_CONTEXT}{user_message}"
    elif additional_context_placement == "user-end":
        user_message = f"{user_message}{SEP_ADDITIONAL_CONTEXT}{additional_context}"

    prompt_formatted = prompt_template.format(
        system_message=system_message, user_message=user_message
    )

    return prompt_formatted
