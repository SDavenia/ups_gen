# Generate answers using the specified model and save valid ones to a .csv file
# The df has name {model_name}.csv and contains columns proposition, prompt, generated_answer, valid, additional_context_key, additional_context_placement
import argparse
import pathlib
import json
import torch
import logging
import re
import pandas as pd

from utils.utils import ensure_reproducibility, prepare_logger, load_model
from utils.run import run_prompts
from utils.data import create_formatted_prompts

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposition_path",
        type=pathlib.Path,
        default=pathlib.Path(
            "../data/political_compass/political_compass_questions.txt"
        ),
        help="Path to the file containing the propositions",
    )
    parser.add_argument(
        "--prompts_path",
        type=pathlib.Path,
        default=pathlib.Path("../data/prompting/prompts_wright.json"),
        help="Path to the JSON file containing the prompts formats.",
    )
    parser.add_argument(
        "--generation_kwargs_path",
        type=pathlib.Path,
        default=pathlib.Path("../data/prompting/generation_args_wright.json"),
        help="Path to the JSON file containing the generation kwargs",
    )
    parser.add_argument(
        "--output_path",
        type=pathlib.Path,
        default=pathlib.Path("../data/generation/")
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF Model ID to use for generation",
    )

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generation"
    )
    parser.add_argument(
        "--additional_context_path",
        type=pathlib.Path,
        default=pathlib.Path("../data/prompting/additional_context.json"),
        help="Path where additional context is placed"
    )

    parser.add_argument(
        "--additional_context_key",
        type=str,
        help="Additional context to be added to the model input before question answering.",
    )
    
    parser.add_argument(
        "--additional_context_placement",
        type=str,
        help="In what part of the prompt the additional context should be specified",
        choices=[None, 'system-beginning', 'system-end', 'user-beginning', 'user-end']
    )

    parser.add_argument(
        "--jailbreak_option",
        type=str,
        help="Option to be used in the jailbreak prompt",
        choices=["jail-01", "jail-02", "jail-03", "jail-04", "jail-05"]
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Seed for reproducibility"
    )
    
    return parser.parse_args()

    
def load_files(proposition_path, prompts_path, generation_kwargs_path):
    # Load PCT propositions
    with open(proposition_path, "r") as f:
        propositions = f.read().splitlines()
    # Load generation templates.
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
    prompts = prompts["open_domain"]

    with open(generation_kwargs_path, "r") as f:
        generation_kwargs = json.load(f)
    return propositions, prompts, generation_kwargs


def log_arguments(args):
    logging.info(f"Model ID: {args.model_id}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Additional Context Key: {args.additional_context_key}")
    logging.info(f"Additional Context Placement: {args.additional_context_placement}")
    logging.info(f"Seed: {args.seed}")


def main():
    # Parse command line arguments, set up logger, device, and enforce reproducibility.
    args = parse_command_line_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = re.match(r".*/(.*)", args.model_id).group(1)

    additional_naming = f"_{args.jailbreak_option}"
    output_file = args.output_path / f"{model_name}{additional_naming}.csv"


    assert args.additional_context_key is None or args.additional_context_placement is not None, "If additional_context_key is specified, additional_context_placement must be specified as well."

    prepare_logger(args)
    log_arguments(args)
    logging.info(f"Device: {device}")
    ensure_reproducibility(args.seed)

    # Read files, prompts and propositions from the PCT
    propositions, prompts, generation_kwargs = load_files(
        args.proposition_path,
        args.prompts_path,
        args.generation_kwargs_path,
    )
    
    # If working in closed domain declare the options
    options = None

    # Load additional context
    if args.additional_context_key is not None:
        logging.info(f"Loading additional context with key {args.additional_context_key}")
        with open(args.additional_context_path, "r") as f:
            additional_context = json.load(f)[args.additional_context_key]
    else:
        additional_context = None
    
    # Create formatted prompts ready to be given as inputs to the model and also return the list of prompt-proposition tuples.
    formatted_prompts, prompt_propositions = create_formatted_prompts(prompts=prompts,
                                                                      propositions=propositions,
                                                                      model_id=args.model_id,
                                                                      jailbreak_option=args.jailbreak_option,
                                                                      additional_context=additional_context,
                                                                      additional_context_placement=args.additional_context_placement
                                                                      )
    logging.info("Successfully formatted prompts")
    
    # Load and prepare model and tokenizer
    model, tokenizer = load_model(args.model_id, device)
    logging.info("Succesfully loaded model")
    # Run the prompts
    model.eval()
    with torch.inference_mode():
        generated_outputs = run_prompts(
            model,
            tokenizer,
            formatted_prompts,
            device,
            batch_size=args.batch_size,
            **generation_kwargs
        )
    
    # Check whether they are valid
    """if args.model_id == "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated":
        valid_outputs = ['valid'] * len(generated_outputs)
    else:
    valid_outputs = [validate_completion(x) for x in generated_outputs]
    logging.info(f"Valid outputs: {len([x for x in valid_outputs if x == 'valid'])} / {len(valid_outputs)}")
    """
    valid_outputs = ['valid'] * len(generated_outputs)
        
    # Save to .csv dataframe with columns proposition, prompt, generated_answer, additional_context_key, additional_context_location
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True)
    prompts_extended, proposition_extended = zip(*prompt_propositions)
    # Also include the generation_kwargs in the rows.
    all_generation_kwargs = [generation_kwargs] * len(proposition_extended)
    all_jailbreak_option = [args.jailbreak_option] * len(proposition_extended)
    output_df = pd.DataFrame({
        "proposition": proposition_extended,
        "prompt": prompts_extended,
        "formatted_prompt": formatted_prompts,
        "generated_answer": generated_outputs,
        "valid": valid_outputs,
        "additional_context_key": args.additional_context_key,
        "additional_context_placement": args.additional_context_placement,
        "jailbreak_option": all_jailbreak_option,
        "generation_kwargs": all_generation_kwargs
    })
    if output_file.exists():
        logging.info(f"{output_file} alread exist: appending to it...")
        output_df.to_csv(output_file, index=False, mode='a', header=False)
    else:
        logging.info(f"{output_file} does not exist exist: creating it...")
        output_df.to_csv(output_file, index=False, mode='w')

if __name__ == "__main__":
    main()
