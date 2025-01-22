#!/bin/bash
#SBATCH -A IscrC_XAI-MRAG
#SBATCH -p boost_usr_prod
#SBATCH --time=0-6:00:00   # Adjust time as needed
#SBATCH --mem=32000
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch_files/evaluate_answers_llama3_small_complete.out
#SBATCH --job-name evaluate_answers

model_data_id="meta-llama/Llama-3.1-8B-Instruct"
batch_size=16
seed=10


source ../../ups_gen_env/bin/activate
source ../../populate_env_vars.sh

python wright_open_to_close.py --model_data_id $model_data_id --batch_size $batch_size --seed $seed --small_complete_run

