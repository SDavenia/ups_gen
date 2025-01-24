import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DECISIONS = ["Strongly Disagree", "Disagree", "None", "Agree", "Strongly Agree"]

def political_compass_base_plot(figsize):
    fig, ax = plt.subplots(figsize=figsize, clip_on=False)
    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    ax.set_xticks(list(range(-10, 11)))
    ax.set_xticklabels([])
    ax.set_yticks(list(range(-10, 11)))
    ax.set_yticklabels([])
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_facecolor('white')
    for sp in ax.spines:
        ax.spines[sp].set_color('#AAAAAA')
        ax.spines[sp].set_visible(True)
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    
    # Define the quadrants with the original color maps
    extent = [0, 10, 0, 10]
    arr = np.array([[1, 1], [1, 1]])
    ax.imshow(arr, extent=extent, cmap='winter', interpolation='none', alpha=0.15)
    
    extent = [-10, 0, 0, 10]
    arr = np.array([[1, 1], [1, 1]])
    ax.imshow(arr, extent=extent, cmap='autumn', interpolation='none', alpha=0.15)
    
    extent = [-10, 0, -10, 0]
    arr = np.array([[1, 1], [1, 1]])
    ax.imshow(arr, extent=extent, cmap='summer', interpolation='none', alpha=0.15)
    
    extent = [0, 10, -10, 0]
    arr = np.array([[1, 1], [1, 1]])
    ax.imshow(arr, extent=extent, cmap='spring_r', interpolation='none', alpha=0.15)
    
    ax.annotate("Economic right", xy=(9.8, -0.75), fontsize=16, ha='right')
    ax.annotate("Economic left", xy=(-9.8, -0.75), fontsize=16)
    ax.annotate("Authoritarian", xy=(0, 8.75), fontsize=16, annotation_clip=False, ha='center')
    ax.annotate("Libertarian", xy=(0, -8.75), fontsize=16, annotation_clip=False, ha='center', va='top')
    
    return fig, ax

def add_datapoints(ax,
                   df,
                   hue_col,
                   style_col,
                   hue_order=None):
    """
    Add datapoints from df to the pct plot. The shape is determined by the model_id and the colour by the prompt.
    """
    markers = ['o', 's', 'D', 'X', 'P']
    colors = sns.color_palette("colorblind", 5)
    
    sns.scatterplot(
        data=df,
        x='economic',
        y='social',
        hue=hue_col,
        style=style_col,
        palette=colors,
        s=130,
        edgecolor='black',
        ax=ax,
        hue_order=hue_order,
        markers=markers,
        alpha=0.9
    )

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

def plot_prompt_robustness(df, sort_by_agreement=True):
    """
    Create a stacked bar chart showing decision distribution for each proposition as percentages.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns 'proposition', 'prompt', and 'decision'
    sort_by_agreement : bool, optional
        If True, sorts propositions by overall agreement percentage (default: True)
    """
    # Ensure decision column is categorical
    decision_order = ['Strongly Disagree', 'Disagree', 'None', 'Agree', 'Strongly Agree']
    
    # Color mapping for decisions
    color_map = {
        'Strongly Disagree': '#8B0000',  # Dark Red (DarkRed)
        'Disagree': '#FF6347',  # Light Red (Tomato)
        'None': '#AAAAAA',  # Gray
        'Agree': '#90EE90',  # Light Green 
        'Strongly Agree': '#006400'  # Dark Green (DarkGreen)
    }
        
    # Group and count decisions for each proposition
    decision_counts = df.groupby('proposition_id').apply(
        lambda x: x['decision'].value_counts(normalize=True) * 100
    ).unstack(fill_value=0)
    
    # Reorder columns to match our predefined order
    decision_percentages = decision_counts.reindex(columns=decision_order, fill_value=0)
    
    # Optional: Sort by agreement if requested
    if sort_by_agreement:
        # Calculate agreement score (weighted average of decisions)
        agreement_score = (
            decision_percentages['Strongly Disagree'] * -2 + 
            decision_percentages['Disagree'] * -1 + 
            decision_percentages['Agree'] * 1 + 
            decision_percentages['Strongly Agree'] * 2
        )
        decision_percentages = decision_percentages.loc[agreement_score.sort_values().index]
    
    # Create the plot
    plt.figure(figsize=(40, 6))
    
    # Create stacked bar plot with percentages
    ax = decision_percentages.plot(kind='bar', stacked=True,
                                   width=1,
                                   edgecolor='white',
                                   linewidth=0.5,
                                   color=[color_map.get(col, '#CCCCCC') for col in decision_percentages.columns])
    
    plt.title('Decision Distribution by Proposition (Percentage)', fontsize=15)
    plt.xlabel('Proposition ID', fontsize=12)
    plt.ylabel('Percentage of Prompts (%)', fontsize=12)
    plt.xticks(fontsize=6, rotation=45)
    plt.ylim(0, 100)
    plt.legend(title='Decision', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('../../data/results_pct/proposition_decisions.png', dpi=300, bbox_inches='tight')

# Example usage
# Uncomment and modify as needed
# sample_df = pd.DataFrame({
#     'proposition': ['Prop1', 'Prop1', 'Prop1', 'Prop2', 'Prop2', 'Prop2'],
#     'prompt': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6'],
#     'decision': ['Agree', 'Disagree', 'Strongly Agree', 'None', 'Disagree', 'Strongly Disagree']
# })
# fig = plot_prompt_robustness(sample_df)
# plt.show()
if __name__ == '__main__':
    # Load the data
    """df = pd.read_csv('../../data/results_pct/pct_results_adjusted.csv')
    print(df)
    all_prompts = df['prompt'].unique().copy()
    all_prompts.sort()
    df['prompt_id'] = df['prompt'].apply(lambda x: np.where(all_prompts == x)[0][0])

    # Create the base plot
    fig, ax = political_compass_base_plot(figsize=(10, 10))
    # Add the data points
    add_datapoints(ax, df, hue_col='prompt_id', style_col='model')

    print("Saving to ../../data/results_pct/pct_results_plot.png")
    plt.savefig('../../data/results_pct/pct_results_plot.png', dpi=300)"""

    model_name = 'Llama-3.1-8B-Instruct'
    df = pd.read_csv(f'../../data/generation_processed/{model_name}.csv')
    df['additional_context_key'] = df['additional_context_key'].fillna('base')
    df['additional_context_placement'] = df['additional_context_placement'].fillna('base')
    
    with open('../../data/label_fixes_wright.json') as f:
        label_fixes = json.load(f)

    with open('../../data/political_compass/political_compass_questions.txt') as f:
        reference_propositions = f.readlines()
    
    # proposition, decision is all we need and we test for prompt robustness
    additional_context_key_filter = 'base'
    additional_context_placement_filter = 'base'

    df_filtered = df[
        (df['additional_context_key'] == additional_context_key_filter) & (df['additional_context_placement'] == additional_context_placement_filter)
    ].copy()
    
    for idx, value in enumerate(df_filtered['decision']):
        # nan is a float in python TODO fix these irregularities.
        if isinstance(value, float):
            df_filtered.loc[idx, 'decision'] = 'None'
        elif value not in DECISIONS:
            value_tomatch = ''.join(e for e in value.lower().strip() if e.isalnum() or e.isspace())
            if value_tomatch in label_fixes.keys():
                # print(f"Changing it to: {label_fixes[value_tomatch]}")
                df_filtered.loc[idx, 'decision'] = label_fixes[value_tomatch]
            else:
                # print("Changing it to: 'None'")
                df_filtered.loc[idx, 'decision'] = 'None'
    
    # Create a new column proposition_id to sort the propositions: TODO ORRIBLE FIX IT
    df_filtered = reorder_propositions(df_filtered, reference_propositions)
    ids = []
    for idx, row in df_filtered.iterrows():
        if idx == 0:
            curr_prop = row['proposition']
            curr_id = 0
            ids.append(curr_id)
        elif row['proposition'] != curr_prop:
            curr_prop = row['proposition']
            curr_id += 1
            ids.append(curr_id)
        else:
            ids.append(curr_id)
    df_filtered['proposition_id'] = ids

    plot_prompt_robustness(df_filtered, sort_by_agreement=False)
