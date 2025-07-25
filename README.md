# ups_gen
This repository contains the code for the paper:[
**Quantifying the Influence of Irrelevant Contexts on Political Opinions Produced by LLMs**](https://aclanthology.org/2025.acl-srw.28/).

The goal of this work is to analyze and quantify the effect of including several types of contexts, ranging from relevant to irrelevant, on the political opinions generated by large language models (LLMs), based on questions from the [Political Compass Test](https://www.politicalcompass.org/) (PCT).


## Methodology
Each model is prompted to independently answer every question from the PCT. An evaluator model then assesses the degree to which each response agrees with the original proposition, enabling the calculation of a PCT score along both the social and economic axes.

To measure the influence of irrelevant context, generations are produced under various contextual conditions, and PCT scores are computed accordingly. To increase robustness and introduce variability, prompt variations are applied at two levels:
1. Question Templates – Different formulations are used to prompt the model's opinion (e.g., "You are writing a blog post on <proposition>" vs. "You are at a political rally and must state your opinion on <proposition>").

2. Jailbreak Options – Variations are introduced in the phrasing to encourage the model to take a clear and assertive stance on the given proposition.

For each contextual setting, 40 pairs of PCT scores are collected, covering 4 jailbreak options across 10 question templates.

## ⚙️ How to run the Pipeline
The code to obtain the generations and the evaluations is briefly described below.

### 1. `generate_answers.py`

Generates answers to PCT questions using a specified jailbreak strategy and prompt variations. Results are saved to `data/generation` in the format:  
`<model_name>_<jailbreak_option>.csv`

**Usage:**
```
python generate_answers.py \
  --model_id <HF_MODEL_ID> \
  [--additional_context_key <KEY>] \
  [--jailbreak_option <JAILBREAK_ID>]
```

Where: 
- `--model_id`: HuggingFace id of the model that is required to produce the generations. 
- `--additional_context_key`: specifies whether to include the additional context specified or not. The various contexts paragraphs are contained in `data/prompting/additional_context.json`. If left empty, the base case without any additional context is used. 
- `--jailreak_option`: specifies which jail-break option to include. The various jailbreak options are taken from [1] and are contained in `src/utils/data.py`.

### 2. `wright_open_to_close.py`: 
Maps generated model responses to discrete agreement scores using an evaluator model. Adds a decision column to the input CSV and stores the result in data/generation_processed in the format: `<model_name>_<jailbreak_option>.csv`. By default (as in [2]), the evaluator is `Mistral-Instruct-7B-v0.3`.

**Usage**:
```
python wright_open_to_close.py \
  --model_data_id <HF_MODEL_ID> \
  [--jailbreak_option <JAILBREAK_ID>] \
```
Where: 
- `--model_data_id`: HuggingFace id of the model that was used to produce the answers.
- `--jailreak_option`: specifies which jail-break option to include.


### 3. `map_to_pct_axis.py`
Given a specific model used to obtain the generations, it takes all files in the directory `data/generation_processed` for that model and computes the PCT score for both economic and social axes for each specific (`additional_context`, `prompt_template`, `jailbreak_option`) tuple and saves them in `data/results_pct/pct_results.csv`. Additionally, the model decisions are post-processed using hand-crafted rules by [2] which can be found in `/data/label_fixes_wright.json`.

**Usage**:
```
python map_to_pct_axis.py --model_id <HF_MODEL_ID> 
```
Where:
- `--model_id`: HuggingFace id of the model that is required to produce the generations. 


## 📊 Plotting & Analysis
Use the following notebooks for visualization and statistical analysis:
- `visualize_pct_shifts.ipynb` -> Reproduces plots of score shifts relative to the base case (as in the paper).
- `visualize_fit_lmm.ipynb` -> Fits linear mixed models for analyzing RQ1 and RQ2.
 
## 📁 Structure of the Repository
- `src/`  
  - `generate_answers.py` – Script for generating answers  
  - `wright_open_to_close.py` – Script for obtaining the evaluator model decisions.
  - `map_to_pct_axis.py` – Maps decision to PCT.
  - `visualize_pct_shifts.ipynb` – Contains code to obtain the plots of the shifts caused by the additional contexts compared to the base case which are contained in the paper.
  - `visualize_fit_lmm.ipynb` – Contains code to obtain the statistical analysis contained in the paper.
  - `utils/`  
    - `data.py` – Contains functions to create the inputs to the models.
    - `run.py` – Contains functions to obtain the model generations.
    - `utils.py` – Contains shared utility functions used throughout the project. 
  - `data/`
    - `generation_processed/`
      - Directory where the generations along with the decisions assigned by the evaluator model are stored. Contains all generations as `.csv` files.
    - `label_fixes_wright.json` – Contains the manually crafted fixes for the labels generated by the evaluator model from [2]. 
    - `political_compass/`
      - `political_compass_questions.txt` – Contains the full set of PCT questions.
    - `prompting/`
      - `additional_context.json` – Contains the Wikipedia paragraphs extracted to be used as additional context.
      - `prompts_wright.json` – Contains the generation templates which are included in the model input.
      - `jailbreak_options_rottger.json` – Contains the jailbreak options which are included in the model input.
      - `generation_args_wright.json` – Contains the args passed to the model for generation (same as [2]).
      - `evaluation_args_wright.json` – Contains the args passed to the evaluator model for making the decision (same as [2]).
    - `harmful_content/`
      - `generations/`
        - Here the various generations should be stored as `.txt` files with a separator `[END]` between each sentence.
      - `extract_hs.ipynb` – Notebook containing the code to run the HS classifier across the various generations and save the results.
      - `visualize_hs_results.ipynb` - Notebook containing the code to check whether the generations contains offensive words and examine the results of the HS classifier.
      - `davidson_hate_words.txt` - List of highly offensive words inspired by the lexicon from [3].

## Cite
To cite this work please cite
```bibtex
@inproceedings{davenia-basile-2025-quantifying,
  title     = "Quantifying the Influence of Irrelevant Contexts on Political Opinions Produced by {LLM}s",
  author    = "D'Avenia, Samuele and Basile, Valerio",
  editor    = "Zhao, Jin and Wang, Mingyang and Liu, Zhu",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)",
  month     = jul,
  year      = "2025",
  address   = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2025.acl-srw.28/",
  pages     = "434--454",
  ISBN      = "979-8-89176-254-1"
}
```


## References
- [1] Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Kirk, Hinrich Schütze, and Dirk Hovy. (2024). **Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in Large Language Models.** *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 15295–15311, Bangkok, Thailand.

- [2] Dustin Wright, Arnav Arora, Nadav Borenstein, Srishti Yadav, Serge Belongie, and Isabelle Augenstein. (2024). **LLM Tropes: Revealing Fine-Grained Values and Opinions in Large Language Models.**  *Findings of the Association for Computational Linguistics: EMNLP 2024*, pp. 17085–17112, Miami, Florida, USA.  Association for Computational Linguistics.

- [3] Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. (2017). **Automated Hate Speech Detection and the Problem of Offensive Language.** *Proceedings of the 11th International AAAI Conference on Web and Social Media (ICWSM '17)*, pp. 512–515, Montreal, Canada.


