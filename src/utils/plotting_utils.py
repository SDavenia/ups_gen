import json
import re
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import reorder_column, fix_label, read_json, read_lines

DECISIONS = ["Strongly Disagree", "Disagree", "None", "Agree", "Strongly Agree"]
DECISIONS_MAP = {
    'Strongly disagree': 0,
    'Strongly Disagree': 0,
    'Disagree': 1,
    'disagree': 1,
    'Agree': 2,
    'agree': 2,
    'Strongly agree': 3,
    'Strongly Agree': 3,
    'None': 1.5
}

COLOR_MAP = {
    'Strongly Disagree': '#8B0000',  # Dark Red (DarkRed)
    'Disagree': '#FF6347',  # Light Red (Tomato)
    'None': '#AAAAAA',  # Gray
    'Agree': '#90EE90',  # Light Gree
    'Strongly Agree': '#006400'  # Dark Green (DarkGreen)
}

def load_plot_pct_scores(pct_results_dir : pathlib.Path = pathlib.Path('../../data/results_pct/'),
                         output_plot_dir : pathlib.Path = pathlib.Path('../../data/results_pct/'),
                         adjusted_scores : bool = True,
                         model_id=None  # By default None and plots for all models are obtained
                         ) -> None:

    df = pd.read_csv(pct_results_dir / f"pct_results{'_adjusted' if adjusted_scores else ''}.csv")
    # If plot for only one model is required, filter the df accordingly
    extra_str = ''
    if model_id is not None:
        model_name = re.match(r".*/(.*)", model_id).group(1)
        # Retain only the rows for the selected model
        df = df[df['model'] == model_name].copy()
        extra_str += '_' + model_name
    
    if adjusted_scores:
        extra_str += '_adjusted'
    
    all_prompts = df['prompt'].unique().copy()
    all_prompts.sort()
    df['prompt_id'] = df['prompt'].apply(lambda x: np.where(all_prompts == x)[0][0])

    # Create the base plot
    fig, ax = political_compass_base_plot(figsize=(10, 10))
    # Add the data points
    add_datapoints(ax, df, hue_col='prompt_id', style_col='model')

    # Save the plot
    if output_plot_dir is not None:     
        output_plot_path = output_plot_dir / f'pct_results_plot{extra_str}.png'
        print(f"Saving to {output_plot_path}")
        plt.savefig(output_plot_path, dpi=300)
    else:
        plt.show()

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
    num_markers = len(df[style_col].unique())
    num_colors = len(df[hue_col].unique())

    """
    # Get all available marker styles in Matplotlib
    all_markers = list(plt.Line2D.filled_markers)  # Only filled markers

    # Remove some redundant or less useful markers
    excluded_markers = [' ', '', None, '|', '_']  # Exclude whitespace and line markers
    markers = [m for m in all_markers if m not in excluded_markers]
    """

    # markers = ['o', 's', 'D', 'X', 'P']
    markers = ['o', 's', 'D', 'X', 'P', '^', 'v', '<', '>', '*', 'H', '8']
    markers = markers[:num_markers]
    colors = sns.color_palette("colorblind", num_colors)
    
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

def load_plot_prompt_robustness(model_id : str,
                                additional_context_key : str,
                                additional_context_placement : str,
                                label_fixes_path : pathlib.Path = pathlib.Path('../../data/label_fixes_wright.json'),
                                proposition_path : pathlib.Path = pathlib.Path('../../data/political_compass/political_compass_questions.txt'),
                                generated_data_path : pathlib.Path = pathlib.Path('../../data/generation_processed/'),
                                output_plot_dir : pathlib.Path = pathlib.Path('../../data/results_pct/'),
                                sort_by_agreement : bool = True,) -> None:
    """
    Given the model_id, additional_context_key, and additional_context_placement, plot the prompt robustness for the model which shows
        the distribution of answers over the various propositions.
    
    Args:
        model_id : str
            The model id for which the prompt robustness is to be plotted.
        additional_context_key : str
            The additional context key for which the prompt robustness is to be plotted.
        additional_context_placement : str
            The additional context placement for which the prompt robustness is to be plotted.
        label_fixes_path : pathlib.Path, optional
            The path to the JSON file containing the label fixes if available (default: ../../data/label_fixes_wright.json)
        proposition_path : pathlib.Path, optional
            The path to the file containing the propositions (default: ../../data/political_compass/political_compass_questions.txt)
        generated_data_path : pathlib.Path, optional
            The path to the directory containing the generated data (default: ../../data/generation_processed/)
        output_plot_dir : pathlib.Path, optional
            The path to the directory where the plot will be saved (default: ../../data/results_pct/)
        sort_by_agreement : bool, optional
            If True, sorts propositions by overall agreement percentage, otherwise the question order is used (default: True)
    """
    model_name = re.match(r".*/(.*)", model_id).group(1)
    label_fixes = read_json(label_fixes_path)
    propositions = read_lines(proposition_path)
    df = pd.read_csv(generated_data_path / f"{model_name}.csv")

    df['additional_context_key'] = df['additional_context_key'].fillna('None')
    df['additional_context_placement'] = df['additional_context_placement'].fillna('None')

    df_filtered = df[
        (df['additional_context_key'] == additional_context_key) & (df['additional_context_placement'] == additional_context_placement)
    ].copy()
    
    # Fix labels
    for idx, value in enumerate(df_filtered['decision']):
        # nan is a float in python TODO fix these irregularities
        if isinstance(value, float):
            df_filtered.loc[idx, 'decision'] = 'None'
        elif value not in DECISIONS:
            fixed_label = fix_label(value, label_fixes)
            df_filtered.loc[idx, 'decision'] = fixed_label
    
    # Create a new column proposition_id to sort the propositions -> id = 0 for first question in PCT, 1 for the next and so on...
    df_filtered = reorder_column(df_filtered, column='proposition', reference_list=propositions, add_id=True)
    
    if sort_by_agreement:
        plot_shaded_bars(df_filtered,
                         column_a='proposition_id',
                         column_b='decision',
                         short_column_a_chars=None,
                         decision_map=DECISIONS_MAP,
                         color_map=COLOR_MAP,
                         output_plot_dir=output_plot_dir)
    else:
        plot_shaded_bars(df_filtered,
                         column_a='proposition_id',
                         column_b='decision',
                         short_column_a_chars=None,
                         # decision_map=DECISIONS_MAP,
                         color_map=COLOR_MAP,
                         output_plot_dir=output_plot_dir)
    
    """plot_prompt_robustness(df_filtered,
                           model_name,
                           additional_context_key,
                           additional_context_placement,
                           output_plot_dir,
                           sort_by_agreement)"""
    

"""
    Create a stacked bar chart showing decision distribution for each proposition as percentages.
    There should be one plot for each (model, additional_context_key, additional_context_placement) combination. So the counts are over the different prompt formats.
    
    Args:
        df : pandas.DataFrame
            DataFrame with columns 'proposition', 'prompt', and 'decision'.
        model_name : str
            The name of the model that is currently being selected, it will be used to save in the correct filename.
        additional_context_key : str
            The additional context key that is currently being selected.
        additional_context_placement : str
            The additional context placement that is currently being selected.
        sort_by_agreement : bool, optional
            If True, sorts propositions by overall agreement percentage, otherwise the question order is used (default: True)
"""
"""
def plot_prompt_robustness(df,
                           model_name : str,
                           additional_context_key : str,
                           additional_context_placement : str,
                           output_plot_dir : pathlib.Path,
                           sort_by_agreement : bool) -> None:
    
    sort_agreement_str = '_agreement' if sort_by_agreement else ''
    output_file = output_plot_dir / f"prompt_robustness_{model_name}_{additional_context_key}_{additional_context_placement}{sort_agreement_str}.png"

    # Ensure decision column is categorical
    decision_order = ['Strongly Disagree', 'Disagree', 'None', 'Agree', 'Strongly Agree']
    
    # Color mapping for decisions
    color_map = {
        'Strongly Disagree': '#8B0000',  # Dark Red (DarkRed)
        'Disagree': '#FF6347',  # Light Red (Tomato)
        'None': '#AAAAAA',  # Gray
        'Agree': '#90EE90',  # Light Gree
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
            decision_percentages['Strongly Disagree'] * -2 + decision_percentages['Disagree'] * -1 + decision_percentages['Agree'] * 1 + decision_percentages['Strongly Agree'] * 2)
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
    print(f"Saving to {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
"""

def plot_shaded_bars(df,
                     column_a,
                     column_b,
                     short_column_a_chars=None,
                     decision_map=None,
                     color_map=None,
                     remove_none=False,
                     show_output=True, 
                     output_plot_dir=None):
    """
    Plots a bar chart for each unique value in column_a, shaded by the percentage
    contribution of each unique value in column_b, with the x-axis ordered
    by the average value in a decision_map.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    column_a (str): Column name for grouping (x-axis).
    column_b (str): Column name for shading (color percentage).
    short_column_a_chars (int, optional): If specified, shortens the values in column_a to
                                        the first or last n characters.
    decision_map (dict, optional): A dictionary mapping unique values in column_b to numeric
                                 values. If provided, the x-axis is ordered by the average
                                 mapped value for each column_a group.
    color_map (dict, optional): A dictionary mapping unique values in column_b to colors.
                               If not provided, default colors will be used.
    remove_none (bool, optional): If True, removes rows where column_b is 'None' for
                                plotting the percentages. Defaults to False.
    show_output (bool, optional): If True, displays the plot. Defaults to True.
    output_plot_dir (pathlib.Path, optional): If specified, saves the plot to the given directory.
    """
    # Copy and preprocess the dataframe
    df = df.copy()
    df.fillna('None', inplace=True)
    df = df[df[column_b] != 'None'].reset_index() if remove_none else df

    # Optionally shorten column_a values
    if short_column_a_chars is not None:
        if short_column_a_chars > 0:
            df[column_a] = [x[:short_column_a_chars] for x in df[column_a]]
        else:
            df[column_a] = [x[short_column_a_chars:] for x in df[column_a]]

    # Calculate counts for each combination of column_a and column_b
    counts = df.groupby([column_a, column_b]).size().reset_index(name='count')

    # Calculate the ordering first if decision_map is provided
    if decision_map:
        # Map column_b values to their decision_map values
        counts['mapped_value'] = counts[column_b].map(decision_map).fillna(0)
        # Calculate weighted average for each group
        weighted_avgs = counts.groupby(column_a).apply(
            lambda x: (x['count'] * x['mapped_value']).sum() / x['count'].sum(), 
            include_groups=False
        )
        # Sort the weighted averages
        ordered_categories = weighted_avgs.sort_values(ascending=True).index
    else:
        ordered_categories = counts[column_a].unique()

    # Ensure column_a is categorical with the correct order
    df[column_a] = pd.Categorical(df[column_a], categories=ordered_categories, ordered=True)
    counts[column_a] = pd.Categorical(counts[column_a], categories=ordered_categories, ordered=True)

    # Pivot the data to calculate percentage contributions
    pivot = counts.pivot(index=column_a, columns=column_b, values='count').fillna(0)
    pivot_percent = pivot.div(pivot.sum(axis=1), axis=0)

    # Ensure the pivot table maintains the correct order
    pivot_percent = pivot_percent.reindex(ordered_categories)

    # Sort columns based on decision_map values if provided
    if decision_map:
        # Create a mapping of column names to their decision_map values
        col_order = {col: decision_map.get(col, float('-inf')) for col in pivot_percent.columns}
        # Sort columns by their decision_map values
        sorted_columns = sorted(pivot_percent.columns, key=lambda x: col_order[x])
        pivot_percent = pivot_percent[sorted_columns]

    # Plot the bars
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(pivot_percent))
    x_ticks = range(len(pivot_percent.index))

    for col in pivot_percent.columns:
        # Use color from color_map if provided, otherwise use default colors
        color = color_map.get(col) if color_map else None
        ax.bar(x_ticks, pivot_percent[col], bottom=bottom, label=str(col), color=color)
        bottom += pivot_percent[col]

    # Add labels and legend
    ax.set_title(f"Shaded Bar Plot by '{column_a}' and '{column_b}'")
    ax.set_xlabel(column_a)
    ax.set_ylabel("Percentage")
    ax.legend(title=column_b, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set x-axis ticks and labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(pivot_percent.index, rotation=45, ha='right')
    
    plt.tight_layout()
    if output_plot_dir:
        print(f"Saving plot to {output_plot_dir / f'shaded_bars_{column_a}_{column_b}.png'}")
        plt.savefig(output_plot_dir / f"shaded_bars_{column_a}_{column_b}.png",)
    elif show_output:
        plt.show()
    else:
        pass

"""
if __name__ == '__main__':
    # Load the data
    # load_plot_prompt_robustness(model_id='meta-llama/Llama-3.1-8B-Instruct',
    #                            additional_context_key='None',
    #                            additional_context_placement='None',
    #                            sort_by_agreement=True)
    
    load_plot_pct_scores(adjusted_scores=True)
"""