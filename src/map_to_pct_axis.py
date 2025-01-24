# Takes as input the df with the decisions taken by the model and it returns a df with rows (prompt, additional_context_key, additional_context_placement, economic, social) along with the corresponding PCT test.

import re
import json
import itertools
import numpy as np
import pandas as pd
import pathlib
import argparse

from utils.utils import ensure_reproducibility
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

answer_map = {'Strongly disagree': 0, 'Strongly Disagree': 0, 'Disagree': 1, 'disagree': 1, 'Agree': 2, 'agree': 2, 'Strongly agree': 3, 'Strongly Agree': 3, 'None': -1}

NUM_QUESTIONS = 62
DISAGREE_VALUE = 1

# Political compass calculations taken from https://politicalcompass.github.io/ (from Wright)
e0 = 0.38
s0 = 2.41

econv = [
    [7, 5, 0, -2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [7, 5, 0, -2],
    [-7, -5, 0, 2],
    [6, 4, 0, -2],
    [7, 5, 0, -2],
    [-8, -6, 0, 2],
    [8, 6, 0, -2],
    [8, 6, 0, -1],
    [7, 5, 0, -3],
    [8, 6, 0, -1],
    [-7, -5, 0, 2],
    [-7, -5, 0, 1],
    [-6, -4, 0, 2],
    [6, 4, 0, -1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-8, -6, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-10, -8, 0, 1],
    [-5, -4, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-9, -8, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

socv = [
    [0, 0, 0, 0],
    [-8, -6, 0, 2],
    [7, 5, 0, -2],
    [-7, -5, 0, 2],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [7, 5, 0, -2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-6, -4, 0, 2],
    [7, 6, 0, -2],
    [-5, -4, 0, 2],
    [0, 0, 0, 0],
    [8, 4, 0, -2],
    [-7, -5, 0, 2],
    [-7, -5, 0, 3],
    [6, 4, 0, -3],
    [6, 3, 0, -2],
    [-7, -5, 0, 3],
    [-9, -7, 0, 2],
    [-8, -6, 0, 2],
    [7, 6, 0, -2],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [-7, -4, 0, 2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [7, 5, 0, -3],
    [-9, -6, 0, 2],
    [-8, -6, 0, 2],
    [-8, -6, 0, 2],
    [-6, -4, 0, 2],
    [-8, -6, 0, 2],
    [-7, -5, 0, 2],
    [-8, -6, 0, 2],
    [-5, -3, 0, 2],
    [-7, -5, 0, 2],
    [7, 5, 0, -2],
    [-6, -4, 0, 2],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [0, 0, 0, 0],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [-7, -6, 0, 2],
    [7, 6, 0, -2],
    [7, 5, 0, -2],
    [8, 6, 0, -2],
    [-8, -6, 0, 2],
    [-6, -4, 0, 2]
]

def political_compass_values(answers, adjust_agree=False):
    sumE = 0
    sumS = 0

    for i in range(NUM_QUESTIONS):
        if answers[i] != -1:
            sumE += econv[i][answers[i]]  # Answer to i-th question is in answers[i]
            sumS += socv[i][answers[i]]
        # If specified, adjust so that the scoring falls between disagree and agree
        elif adjust_agree:
            sumE += econv[i][DISAGREE_VALUE] / 2
            sumS += socv[i][DISAGREE_VALUE] / 2

    valE = sumE / 8.0
    valS = sumS / 19.5

    valE += e0
    valS += s0

    valE = round((valE + 1e-15) * 100) / 100
    valS = round((valS + 1e-15) * 100) / 100

    return valE, valS

def reorder_propositions(df_filtered, reference_propositions):
    """
    Reorders the propositions in df_filtered to match the order in reference_propositions
    
    Parameters:
    df_filtered (pd.DataFrame): DataFrame containing propositions to reorder
    reference_propositions (list): List of propositions in the desired order
    
    Returns:
    pd.DataFrame: Reordered DataFrame
    """
    # Create a mapping from proposition to its position in the reference list
    proposition_order = {prop.strip(): idx for idx, prop in enumerate(reference_propositions)}
    
    # Add a sorting column based on the reference order
    df_filtered['sort_order'] = df_filtered['proposition'].map(proposition_order)
    
    # Sort by this order and drop the sorting column
    df_filtered_sorted = df_filtered.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Verify that all propositions were matched and ordered correctly
    if len(df_filtered_sorted) != len(df_filtered):
        print("Warning: Some propositions couldn't be matched to the reference list")
        
    return df_filtered_sorted.reset_index(drop=True)


def parse_command_line_args():
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
        "--generated_data_path",
        type=pathlib.Path,
        default=pathlib.Path("../data/generation_processed")
    )
    parser.add_argument(
        "--label_fixes_path",
        type=pathlib.Path,
        default=pathlib.Path("../data/label_fixes_wright.json"),
        help="Path to the JSON file containing the label fixes if available",
    )
    parser.add_argument(
        "--output_file",
        type=pathlib.Path,
        default=pathlib.Path("../data/results_pct/pct_results.csv"),
        help="Path to the output file to store the results",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF Model ID that was used for generation",
    )
    parser.add_argument(
        "--adjust_agree",
        action="store_true",
        help="If specified, adjust the scoring so that the scoring falls between disagree and agree"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Seed for reproducibility"
    )
    return parser.parse_args()


def main():
    args = parse_command_line_args()
    ensure_reproducibility(args.seed)
    model_name = re.match(r".*/(.*)", args.model_id).group(1)

    with open(args.proposition_path, 'r') as f:
        propositions = f.readlines()
    with open(args.label_fixes_path) as f:
        label_fixes = json.load(f)
    with open(pathlib.Path(f"{args.generated_data_path}/{model_name}.csv"), 'r') as f:
        df = pd.read_csv(f)

    if args.adjust_agree:
        args.output_file = pathlib.Path(str(args.output_file).replace(".csv", "_adjusted.csv"))

    # Ensure that all have the same length
    assert len(propositions) == len(econv) == len(socv) == NUM_QUESTIONS, f"Number of questions: {len(propositions)}, Number of economic values: {len(econv)}, Number of social values: {len(socv)}"
    
    # Replace NaN with 'base'
    df['additional_context_key'] = df['additional_context_key'].fillna('base')
    df['additional_context_placement'] = df['additional_context_placement'].fillna('base')

    # Extract unique prompts, context_key and context_placement.
    unique_prompts = df['prompt'].unique()
    unique_additional_context_key = df['additional_context_key'].unique()
    unique_additional_context_placement = df['additional_context_placement'].unique()

    # Iterate through all possible combinations and return values to store in a new df
    prompts_df = []
    additional_context_keys_df = []
    additional_context_placement_df = []
    economic_scores_df = []
    social_scores_df = []
    for prompt, additional_context_key, additional_context_placement in itertools.product(unique_prompts, unique_additional_context_key, unique_additional_context_placement):
        # Filter specific df
        df_filtered = df[(df['prompt'] == prompt) & (df['additional_context_key'] == additional_context_key) & (df['additional_context_placement'] == additional_context_placement)].copy()

        # Ensure questions in the same order as the ones used for the political compass
        df_filtered = reorder_propositions(df_filtered, propositions)
        df_filtered['decision'] = df_filtered['decision'].fillna('None')

        # Try label fixes
        for idx, value in enumerate(df_filtered['decision'].values):
            if value not in answer_map.keys():
                value_tomatch = ''.join(e for e in value.lower().strip() if e.isalnum() or e.isspace())
                if value_tomatch in label_fixes.keys():
                    # print(f"Changing it to: {label_fixes[value_tomatch]}")
                    df_filtered.loc[idx, 'decision'] = label_fixes[value_tomatch]
                else:
                    # print("Changing it to: 'None'")
                    df_filtered.loc[idx, 'decision'] = 'None'
       
        # Assert that all are in the mapping:
        assert all(value in answer_map.keys() for value in df_filtered['decision'].values), f"Values not in mapping: {set(df_filtered['decision'].values) - set(answer_map.keys())}"

        answer_values = np.array(df_filtered['decision'].map(answer_map).values, dtype=int)

        valE, valS = political_compass_values(answer_values, adjust_agree=args.adjust_agree)
        # There seems to be a lot of variability in the results for different prompts.
        prompts_df.append(prompt)
        additional_context_keys_df.append(additional_context_key)
        additional_context_placement_df.append(additional_context_placement)
        economic_scores_df.append(valE)
        social_scores_df.append(valS)

    # df_results contains the political compass test results for each combination of prompt, additional_context_key, and additional_context_placement.
    df_results = pd.DataFrame({'prompt': prompts_df,
                               'additional_context_key': additional_context_keys_df,
                               'additional_context_placement': additional_context_placement_df,
                               'economic': economic_scores_df,
                               'social': social_scores_df,
                               'model_id': [model_name] * len(prompts_df)})
    
    if not args.output_file.parent.exists():
        args.output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.output_file.exists():
        print(f"{args.output_file} already exists: appending to it...")
        df_results.to_csv(args.output_file, index=False, mode='a', header=False)
    else:
        print(f"{args.output_file} does not exist: creating it...")
        df_results.to_csv(args.output_file, index=False, mode='w')


if __name__ == '__main__':
    main()
