#!/bin/bash
#SBATCH --account=def-umaivodj

#SBATCH --time=00:50:00 #Good for 100_000 samples from ACS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G

#SBATCH --gres=gpu:1


echo "Starting job on $(hostname)"

# 1. Load required system modules BEFORE activating the environment
module load python/3.11.5
source ../../new_venv/bin/activate


NAME="$1"
MODEL_PATH="$2"
DATA_PATH="$3"
RAW_DATA_PATH="$4"

echo "Launching encoding for: $NAME from $MODEL_PATH"
# python single_embedding.py --name_model "$NAME" --model "$MODEL_PATH"
python run_embeddings/single_embedding.py --name_model "$NAME" --model "$MODEL_PATH" --data_path "$DATA_PATH" --raw_data_path "$RAW_DATA_PATH"


# 4. Deactivate the environment (good practice)
deactivate

echo "Job completed."