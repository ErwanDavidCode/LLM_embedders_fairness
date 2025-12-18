#!/bin/bash
#SBATCH --account=def-umaivodj
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --array=0-0


# Variables from pipeline
SEED="$SEED"
KVAL="$KVAL"
OUTPUT_DIR="$1"

module load python/3.11.5
source ../../new_venv/bin/activate

# Embeders list
embedder_names=($(python -c "import json; print(' '.join(json.load(open('embedders.json')).keys()))"))

# get the current embedder from the array
NAME=${embedder_names[$SLURM_ARRAY_TASK_ID]}
INPUT="${DATA_PATH}/${NAME}.csv"

echo "Running single_score_supervised_fairentropy.py for $NAME ($INPUT)"

python -m emir.emir.single_score_supervised_fairentropy \
    --input1 "$INPUT" --name1 "$NAME" \
    --output "$OUTPUT_DIR" \
    --seed "$SEED" --k "$KVAL"

echo "Job finished for $NAME."
deactivate
