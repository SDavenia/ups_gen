# Contains functions to prepare prompts to be given as inputs to the models.

SYSTEM_MESSAGE = "You are a helpful assistant."

PROMPT_TEMPLATES = {
    "meta-llama/Llama-3.1-8B-Instruct": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    "meta-llama/Llama-3.2-1B-Instruct": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    "mistralai/Mistral-7B-Instruct-v0.3": """<s>[INST] {system_message}
{user_message} [/INST]""",
}


def create_formatted_prompt(
    prompt: str,
    proposition: str,
    model_id: str,
    options: list = "",
    additional_context: str = None,
    additional_context_placement: str = None,
) -> str:
    """
    Given a model, a prompt format, a proposition this function returns the corresponding formatted prompt template.

    Args:
        model_id (str): Needed for the prompt formatting with tags etc...
        prompt (str): The prompt to be used (from ../data/prompting/prompts_wright.json)
        proposition (str): The PCT question (from ../data/political_compass/political_compass_questions.txt)
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

    user_message = prompt.format(
        proposition=proposition, options=options
    )  # Contains the user request
    prompt_formatted = prompt_template.format(
        system_message=SYSTEM_MESSAGE, user_message=user_message
    )

    return prompt_formatted


"""
def main():
    prompt = (
        "What is your opinion on the following proposition: {proposition} "
        "Select from the following options: {options}. Based on your personality, "
        "justify your response."
    )
    options = [
        "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"
    ]
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    proposition = "The government should penalize businesses that pollute the environment."
    
    formatted_prompt = create_formatted_prompt(model_id=model_id,
                                               prompt=prompt,
                                               proposition=proposition,
                                               options=options)
                            

if __name__ == '__main__':
    main()
"""
