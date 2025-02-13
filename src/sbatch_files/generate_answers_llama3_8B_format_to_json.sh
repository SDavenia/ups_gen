#!/bin/bash
#SBATCH -A IscrC_XAI-MRAG
#SBATCH -p boost_usr_prod
#SBATCH --time=0-23:00:00
#SBATCH --mem=32000
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch_files/output/generate_answers_llama3_8B_format_to_json.out
#SBATCH --job-name generate_answers_llama3_8B_format_to_json

model_id="meta-llama/Llama-3.1-8B-Instruct"
batch_size=32
seed=10

source ../../ups_gen_env/bin/activate
source ../../populate_env_vars.sh

# Base generation
echo "Generating answer with base model"
python generate_answers.py --model_id $model_id --batch_size $batch_size --seed $seed --format_to_json

# Generation with additional context -> need to specify additional context key and position
for add_context_key in "wiki_mus_classical" "wiki_mus_heavy-metal" "wiki_mus_jazz" "wiki_mus_hip-pop" "wiki_mus_rock" "wiki_mus_pop" "wiki_obj_table" "wiki_obj_chair" "wiki_obj_sink" "wiki_pol_trump" "wiki_pol_obama" "wiki_pol_biden" "wiki_pol_bush"; do
    for add_context_pos in "system-beginning" "system-end" "user-beginning" "user-end"; do
        echo "Generating answer with additional context key: $add_context_key and position: $add_context_pos"
        python generate_answers.py --model_id $model_id --batch_size $batch_size --seed $seed --additional_context_key $add_context_key --additional_context_placement $add_context_pos --format_to_json
    done
done