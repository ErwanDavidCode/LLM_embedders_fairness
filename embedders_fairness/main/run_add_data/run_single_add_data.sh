#!/bin/bash
#SBATCH --account=def-umaivodj
#SBATCH --job-name=run_single_clustering
#SBATCH --time=01:00:00 # Good for 100_000 samples from ACS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G


echo "Job starting on $(hostname)"

# Charger l'environnement
module load python/3.11.5
source ../../new_venv/bin/activate

# Lancer le script
INPUT="$1"
NAME="$2"
RAW_DATA_PATH="$3"

echo "Adding sensitive attributes for $INPUT for $NAME"
python run_add_data/single_add_data_to_embeddings.py --input "$INPUT" --raw_data_path "$RAW_DATA_PATH" --model_name "$NAME"

# Désactiver l'environnement
deactivate

echo "Job completed."

