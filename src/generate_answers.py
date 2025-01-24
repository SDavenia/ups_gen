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
from utils.rottger_et_al_helpers import validate_completion


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
        "--closed_domain",
        action="store_true",
        help="Whether to use the closed form of the prompts. Defaults to False",
    )

    parser.add_argument(
        "--format_to_json",
        action="store_true",
        help="If true, the model is prompted to produce the output in JSON format. Defaults to False",
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
        choices=[None, 'wiki_classical', 'wiki_heavy-metal', 'wiki_jazz', 'wiki_hip-pop', 'wiki-rock', 'wiki_pop']
    )
    
    parser.add_argument(
        "--additional_context_placement",
        type=str,
        help="In what part of the prompt the additional context should be specified",
        choices=[None, 'system-beginning', 'system-end', 'user-beginning', 'user-end']
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Seed for reproducibility"
    )

    # TESTING
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to use the test mode. Defaults to False",
    )
    parser.add_argument(
        "--test_large",
        action="store_true",
        help="Whether to use the test mode with larger number of propositions. for testing on bigger machine"
    )
    parser.add_argument(
        "--small_complete_run",
        action="store_true",
        help="Whether to run a small complete run"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Whether to save the outputs to a .csv file. Defaults to False"
    )
    return parser.parse_args()

    
def load_files(proposition_path, prompts_path, generation_kwargs_path, closed_domain):
    with open(proposition_path, "r") as f:
        propositions = f.read().splitlines()
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
        if closed_domain:
            prompts = prompts["closed_domain"]
        else:
            prompts = prompts["open_domain"]
    with open(generation_kwargs_path, "r") as f:
        generation_kwargs = json.load(f)
    return propositions, prompts, generation_kwargs


def log_arguments(args):
    logging.info(f"Model ID: {args.model_id}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Closed Domain questions: {args.closed_domain}")
    logging.info(f"Format Output to JSON: {args.format_to_json}")
    logging.info(f"Additional Context Key: {args.additional_context_key}")
    logging.info(f"Additional Context Placement: {args.additional_context_placement}")
    logging.info(f"Seed: {args.seed}")


def main():
    # Parse command line arguments, set up logger, device, and enforce reproducibility.
    args = parse_command_line_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = re.match(r".*/(.*)", args.model_id).group(1)
    
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
        args.closed_domain,
    )
    # For testing purposes only keep the first 2 propositions and prompts.
    if args.test:
        propositions = propositions[:2]
        prompts = prompts[:1]
        generation_kwargs = {"max_new_tokens": 20, "temperature": 0.6, "do_sample": True}
    
    if args.test_large:
        propositions = propositions[:10]
        prompts = prompts[:5]
        generation_kwargs = {"max_new_tokens": 100, "temperature": 0.6, "do_sample": True}
    
    if args.small_complete_run:
        propositions = propositions
        prompts = prompts[:2]
        # generation_kwargs is the default one used in Wright.

    # If working in closed domain declare the options
    if args.closed_domain:
        options = ['Strongly disagree', 'Disagree', 'Agree', 'Strongly agree']
    else:
        options = None

    # Load additional context
    if args.additional_context_key is not None:
        logging.info(f"Loading additional context with key {args.additional_context_key}")
        with open(args.additional_context_path, "r") as f:
            additional_context = json.load(f)[args.additional_context_key]
            if args.test:
                additional_context = additional_context[:25]
    else:
        additional_context = None
    
    # Create formatted prompts ready to be given as inputs to the model and also return the list of prompt-proposition tuples.
    formatted_prompts, prompt_propositions = create_formatted_prompts(prompts=prompts,
                                                                      propositions=propositions,
                                                                      model_id=args.model_id,
                                                                      format_to_json=args.format_to_json,
                                                                      options=options,
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
    if args.test:
        logging.info(f"Generated outputs: {[repr(x) for x in generated_outputs]}")
    
    # Check whether they are valid
    valid_outputs = [validate_completion(x) for x in generated_outputs]
    if args.test:
        for i in range(len(valid_outputs)):
            print(generated_outputs[i])
            print(f"{'valid' if valid_outputs[i] == 'valid' else 'invalid'}")
            print(f"{' - ' * 20}")
    logging.info(f"Valid outputs: {len([x for x in valid_outputs if x == 'valid'])} / {len(valid_outputs)}")
    
    # If for testing no save -> return
    if args.no_save:
        return
    
    # Save to .csv dataframe with columns proposition, prompt, generated_answer, additional_context_key, additional_context_location
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True)
    prompts_extended, proposition_extended = zip(*prompt_propositions)
    # Also include the generation_kwargs in the rows.
    all_generation_kwargs = [generation_kwargs] * len(proposition_extended)
    output_df = pd.DataFrame({
        "proposition": proposition_extended,
        "prompt": prompts_extended,
        "formatted_prompt": formatted_prompts,
        "generated_answer": generated_outputs,
        "valid": valid_outputs,
        "additional_context_key": args.additional_context_key,
        "additional_context_placement": args.additional_context_placement,
        "generation_kwargs": all_generation_kwargs
    })
    if args.test:
        print(output_df.head())
    # If test include it in the name.
    additional_naming = "_test" if args.test else "_test_large" if args.test_large else "_small_complete_run" if args.small_complete_run else ""
    # additional_naming = "_test" if args.test else "_test_large" if args.test_large else ""
    output_file = args.output_path / f"{model_name}{additional_naming}.csv"
    if output_file.exists():
        logging.info(f"{output_file} alread exist: appending to it...")
        output_df.to_csv(output_file, index=False, mode='a', header=False)
    else:
        logging.info(f"{output_file} does not exist exist: creating it...")
        output_df.to_csv(output_file, index=False, mode='w')

if __name__ == "__main__":
    main()
