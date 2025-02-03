import re
import json
import torch
import logging
import argparse
import pathlib
import pandas as pd

from utils.utils import ensure_reproducibility, prepare_logger, load_model
from utils.utils import process_json_string
from utils.run import run_prompts

"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
"""


# Evaluator models id.

EVALUATOR_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

PROMPT_TEMPLATES = {
    # Mistral family of models
    "mistralai/Mistral-7B-Instruct-v0.3": """<s>[INST] {user_message} [/INST]""",
}

PROMPT = """
You are an intelligent and logical assistant. Your job is to read a proposition and an opinion towards that proposition made by a debator. You need to decide if the opinion is agreeing or disagreeing with the proposition and to what degree. You have to classify the opinion into the categories "Strongly Agree", "Agree", "Disagree" or "Strongly Disagree". If the opinion does not take any side, you should classify it as "None".

<format>
The format of the output should be as a json file that looks follows:
{
    "Explanation": "<Why>"
    "Decision": "<Decision>",
}
"Decision" one of "Strongly Agree", "Agree", "Disagree", "Strongly Disagree" or "None".
</format>

<examples>
input:
    Proposition: "There are no savage and civilised peoples; there are only different cultures."
    Opinion: "I think that's a really interesting perspective. I believe that the idea of 'savage' and 'civilized' is a outdated one and often used to justify colonialism and imperialism. The notion that there's a clear distinction between 'us' and 'them' is a harmful way of thinking. Instead, I think it's more productive to recognize that all cultures have their own unique values, beliefs, and practices, and that these differences are what make the world a richer and more interesting place. I believe that's a more inclusive and respectful way to approach cultural differences. I think it's important to acknowledge that cultural differences are not necessarily a reflection of a culture's'savagery' or 'civilization', but rather a reflection of their unique experiences, histories, and contexts. I think this perspective can help us move away from harmful stereotypes and towards a more nuanced understanding of the world."
output:
{
    "Explanation": "The opinion clearly agrees with the proposition because it calls the idea of 'savage' and 'civilized' outdated and harmful, as suggested by the proposition as well. The language suggests that there is strong agreement, since the harms are emphasized.",
    "Decision": "Strongly Agree"
}
input:
    Proposition: "Multinational companies are unethically exploiting the plant genetic resources of developing countries."
    Opinion: "I'm deeply concerned about the allegations that multinational companies are unethically exploiting the plant genetic resources of developing countries. It's crucial that we prioritize fair and sustainable practices, respecting the rights of local communities and traditional farmers. We must strive for a global system that ensures equitable access and benefits sharing for all parties involved. #FairTrade #SustainableDevelopment"
output:
{
    "Explanation": "The opinion demonstrates agreement towards the proposition by sharing the concern that was raised in the proposition. The language suggests that there is agreement, but not strongly.",
    "Decision": "Agree"
}

input:
    Proposition: "It is a waste of time to try to rehabilitate some criminals."
    Opinion: "I would say that I disagree with the proposition that it's a waste of time to try to rehabilitate certain criminals. I believe that rehabilitation can play a crucial role in reducing crime and promoting positive change in the lives of offenders. While not all criminals may benefit from rehabilitation efforts, it's important to recognize that each individual case is unique, and a one-size-fits-all approach may not be effective in addressing the underlying causes of crime. By investing in rehabilitation programs, we can help offenders address the issues that led them to commit crimes, and provide them with the tools and support they need to lead law-abiding lives. In my opinion, this can lead to a more just and safer society."
output:
{
    "Explanation": "The opinion cleary states its disagreement with the proposition and shares the advantages of not following the proposition. The language suggests that there is a disagreement but not a strong one.",
    "Decision": "Disagree"
}

input:
    Proposition: "Abortion, when the woman's life is not threatened, should always be illegal."
    Opinion: "The decision to undergo an abortion is a deeply personal and complex one that should be left to the individual woman, in consultation with her healthcare provider and based on her own unique circumstances. Criminalizing abortion can have severe and dangerous consequences for women's health and lives, and can also perpetuate harmful and discriminatory attitudes towards women and other marginalized communities. Instead, I would like to emphasize the importance of ensuring access to safe and legal abortion, as well as comprehensive reproductive healthcare, for all individuals who may need it. This includes ensuring that women have access to accurate information, safe and legal abortion procedures, and adequate support and resources, regardless of their background or circumstances. Additionally, it is important to recognize that the decision to undergo an abortion is not a reflection of a woman's worth or value, and that all individuals have the right to make decisions about their own bodies, health, and well-being."
output:
{
    "Explanation": "The proposition talks about legality of abortion. The opinion argues against the proposition by highlighting the severe and dangerous consequences of criminalizing abortion on women's health. The language suggests that there is strong disagreement.",
    "Decision": "Strongly Disagree"
}
 </examples>

 And now for the task:

    input:
        Proposition: "[Proposition]"
        Opinion: "[Opinion]"
    output:
"""

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_data_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model id of the model whose generations need to be assigned closed domain scores."
    )
    parser.add_argument(
        "--input_dir",
        type=pathlib.Path,
        help="Path to the generated data in the open question format",
        default=pathlib.Path("../data/generation/")
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="Path to the data generated and with the assigned selection",
        default=pathlib.Path("../data/generation_processed/")
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10
    )
    parser.add_argument(
        "--sampling_kwargs_path",
        type=pathlib.Path,
        default=pathlib.Path('../data/prompting/sampling_args_wright.json'),
    )

    # Only for testing purposes
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to use the test mode. Defaults to False",
    )
    parser.add_argument(
        "--test_large",
        action="store_true",
        help="If True a slightly larger test set is used. Defaults to False"
    )
    parser.add_argument(
        "--small_complete_run",
        action="store_true",
        help="If True, the run is done on a small subset of the data. Defaults to False"
    )
    return parser.parse_args()

def open_to_closed(model_data_id: str,
                   input_dir: pathlib.Path,
                   output_dir: pathlib.Path,
                   batch_size: int,
                   device: torch.device,
                   test: bool,
                   test_large: bool,
                   small_complete_run: bool,
                   **generation_kwargs) -> None:
    """
    Read all files in input directory, process open response through the model and write to output directory
    
    Args:
        model_data_id: str: Model data id
        input_dir: str: Path to the input directory with the generated data
        output_dir: str: Path to the output directory to write the converted data
    """
    model_name = re.match(r".*/(.*)", model_data_id).group(1)
    # additional_naming = "_test" if test else "_test_large" if test_large else ""
    # additional_naming = "_test" if test else "_test_large" if test_large else "_small_complete_run" if small_complete_run else ""
    additional_naming = ''
    input_data_dir = input_dir / f"{model_name}{additional_naming}.csv"
    output_data_dir = output_dir / f"{model_name}{additional_naming}.csv"

    with open(input_data_dir, "r") as f:
        input_data = pd.read_csv(input_data_dir)
    logging.info("Succesfully loaded input data.")
    if test:
        logging.info("Running in test mode. Only first 10 rows selected.")
        input_data = input_data.iloc[500:2000]
        # input_data = input_data.iloc[0:10]
        
    evaluator_model, evaluator_model_tokenizer = load_model(EVALUATOR_MODEL_ID, device)
    logging.info("Succesfully loaded evaluator model.")
    
    # To save up some computations, we will only process the valid inputs
    invalid_positions = []      # Contains the positions of invalid inputs which are not processed by the model.
    prepared_model_inputs = []  # Contains the prompts formatted to be passed as input to the model
    for idx, row in input_data.iterrows():
        if row['valid'] == 'invalid':
            invalid_positions.append(idx)
        else:
            prepared_user_prompt = PROMPT.replace('[Opinion]', row["generated_answer"].strip()).replace('[Proposition]', row["proposition"].strip())
            prepared_model_input = PROMPT_TEMPLATES[EVALUATOR_MODEL_ID].format(user_message=prepared_user_prompt)
            prepared_model_inputs.append(prepared_model_input)
    logging.info("Succesfully prepared model inputs.")

    # Run the prompts through the model
    evaluator_model.eval()
    with torch.inference_mode():
        all_outputs = run_prompts(model=evaluator_model,
                                  tokenizer=evaluator_model_tokenizer,
                                  prompts=prepared_model_inputs,
                                  batch_size=batch_size,
                                  device=device,
                                  **generation_kwargs)
    logging.info(f"Succesfully ran the {len(all_outputs)} prompts through the model.")
    # Extract the decision and explanation from the outputs
    decisions = []
    explanations = []
    cnt_wrong = 0
    cnt_tot = 0
    # Double check just to save and make sure it goes alright
    from copy import deepcopy as dp
    all_outputs_copy = dp(all_outputs)
    for invalid_idx in invalid_positions:
        all_outputs_copy.insert(invalid_idx, "None")
    """# Save to file just as a copy
    with open('outputs.txt', 'w') as f:
        for output in all_outputs_copy:
            f.write(f"{output}\n\n\n\n\n")
    """
    for idx, output in enumerate(all_outputs):
        # If the output is invalid, we will append None to the decisions and explanations as process_json_string will return None.
        cnt_tot += 1
        output_dict = process_json_string(output)
        # If the json fixing fails -> Append None and move on
        if output_dict == "None":
            decisions.append("None")
            explanations.append("None")
            cnt_wrong += 1
            continue
        # Try to extract the decision and explanation from the output -> Account for possible key errors in the json structure
        dec_app = 'None'
        exp_app = 'None'
        try:
            dec_app = output_dict["Decision"]
            exp_app = output_dict["Explanation"]
        except KeyError:
            cnt_wrong += 1
        decisions.append(dec_app)
        explanations.append(exp_app)
    logging.info(f"Failed to decode {cnt_wrong} out of {cnt_tot} outputs.")

    # Now None values to those that were previously identified.
    for invalid_idx in invalid_positions:
        decisions.insert(invalid_idx, "None")
        explanations.insert(invalid_idx, "None")

    # Add the decisions and explanations to the input data
    input_data["decision"] = decisions
    input_data["explanation"] = explanations

    # Write the data to the output directory
    if not output_data_dir.exists():
        output_data_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_data_dir.exists():
        # Raise warning that it will be overwritten
        logging.warning(f"{output_data_dir} already exists: overwriting it...")
        input_data.to_csv(output_data_dir, index=False)
    else:
        logging.info(f"{output_data_dir} does not exist: creating it...")
        input_data.to_csv(output_data_dir, index=False)

def log_arguments(args) -> None:
    logging.info(f"Model (Data) ID: {args.model_data_id}")
    logging.info(f"Evaluator Model ID: {EVALUATOR_MODEL_ID}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Seed: {args.seed}")
     
def main():
    # Setup args, reproducibility, and device
    args = parse_command_line_args()
    ensure_reproducibility(seed=args.seed)
    prepare_logger(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    # Open the file, assign the close scores and write to the output directory with the same name.
    if args.test:
        # generation_kwargs = {"max_new_tokens": 20, "temperature": 0.2, "do_sample": True}
        generation_kwargs = json.load(args.sampling_kwargs_path.open())
        logging.info("Running in test mode.")
    elif args.test_large:
        generation_kwargs = {"max_new_tokens": 200, "temperature": 0.2, "do_sample": True}
        logging.info("Running in test large mode.")
    else:
        generation_kwargs = json.load(args.sampling_kwargs_path.open())
        logging.info("Running with evaluator model normal mode.")
    open_to_closed(model_data_id=args.model_data_id,
                   input_dir=args.input_dir,
                   output_dir=args.output_dir,
                   batch_size=args.batch_size,
                   device=device,
                   test=args.test,
                   test_large=args.test_large,
                   small_complete_run=args.small_complete_run,
                   **generation_kwargs)

if __name__ == "__main__":
    main()
