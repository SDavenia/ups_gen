# Generate answers using the specified model and save valid ones to a .csv file
# The df has name {model_name}.csv and contains columns proposition, prompt, generated_answer, valid, additional_context_key
import argparse
import pathlib
import json
import torch
import logging
import re
import pandas as pd

from utils.utils import ensure_reproducibility, prepare_logger, load_model
from utils.data import create_formatted_prompts
from utils.run import run_prompts

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        help="HF Model ID to use for generation",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for generation"
    )

    # Directories: should be left as default for compatibility with the other scripts.
    parser.add_argument(
        "--additional_context_path",
        type=pathlib.Path,
        default=pathlib.Path("../data/prompting/additional_context.json"),
        help="Path where additional context is placed"
    )
    parser.add_argument(
        "--jailbreak_options_path",
        type=pathlib.Path,
        default = pathlib.Path("../data/prompting/jailbreak_options_rottger.json")
    )
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
    logging.info(f"Seed: {args.seed}")


def main():
    # Parse command line arguments, set up logger, device, and enforce reproducibility.
    args = parse_command_line_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = re.match(r".*/(.*)", args.model_id).group(1)
    output_file = args.output_path / f"{model_name}.csv"
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True)

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

    # Load additional contexts
    with open(args.additional_context_path, "r") as f:
        additional_context_dict = json.load(f)
    additional_context_dict['base'] = None

    # Load jailbreak options
    with open(args.jailbreak_options_path, "r") as f:
        jailbreak_options = json.load(f)

    # Load and prepare model and tokenizer
    model, tokenizer = load_model(args.model_id, device)
    logging.info("Succesfully loaded model")

    # Iterate and generate over jailbreak options and additional contexts
    output_df = pd.DataFrame()
    for jailbreak_key, jailbreak_option in jailbreak_options.items():
        for additional_context_key, additional_context in additional_context_dict.items():
            logging.info(f"Jailbreak option: {jailbreak_option}")
            logging.info(f"Additional context key: {additional_context_key}")    
            # Create formatted prompts ready to be given as inputs to the model and also return the list of prompt-proposition tuples.
            formatted_prompts, prompt_propositions = create_formatted_prompts(prompts=prompts,
                                                                              propositions=propositions,
                                                                              model_id=args.model_id,
                                                                              jailbreak_option=jailbreak_option,
                                                                              additional_context=additional_context,
                                                                              )
            logging.info("\tSuccessfully formatted prompts")
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
            logging.info("\tSuccessfully generated outputs")
            # Prepare all necessary variables and save to a dataframe
            prompts_extended, proposition_extended = zip(*prompt_propositions)
            all_generation_kwargs = [generation_kwargs] * len(proposition_extended)
            all_jailbreak_option = [jailbreak_key] * len(proposition_extended)
            output_df_temp = pd.DataFrame({
                "proposition": proposition_extended,
                "prompt": prompts_extended,
                "formatted_prompt": formatted_prompts,
                "generated_answer": generated_outputs,
                "additional_context_key": additional_context_key,
                "jailbreak_option": all_jailbreak_option,
                "generation_kwargs": all_generation_kwargs
            })
            # Concatenate the new dataframe with the previous one
            output_df = pd.concat([output_df, output_df_temp], ignore_index=True)
    logging.info(f"Saving generated answers to {output_file}")
    output_df.to_csv(output_file, index=False, mode='w')

if __name__ == "__main__":
    main()
