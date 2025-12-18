#!/bin/bash
#SBATCH --account=def-umaivodj

#SBATCH --time=01:00:00  #00:30:00 #time out for UAE large - 15 embeders
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G #32G

#SBATCH --array=0-0 #Will be replaced as array is an argument specified in run_all_steps_different_k

# Variables from pipeline
SEED="$SEED"
KVAL="$KVAL"
OUTPUT_DIR="$1"


module load python/3.11.5
source ../../new_venv/bin/activate

# Read embedders list
embedder_names=($(python -c "import json; print(' '.join(json.load(open('embedders.json')).keys()))"))


# Select reference embedder from array ID
NAME1=${embedder_names[$SLURM_ARRAY_TASK_ID]}
INPUT1="${DATA_PATH}/${NAME1}.csv"


echo "Reference embedder for single_score_supervised_fairis.py: $NAME1 ($INPUT1)"

# Compare reference with all others
for NAME2 in "${embedder_names[@]}"; do
    INPUT2="${DATA_PATH}/${NAME2}.csv"

    echo "Running single_score_supervised_fairis.py for $NAME1 vs $NAME2"
    python -m emir.emir.single_score_supervised_fairis \
        --input1 "$INPUT1" --name1 "$NAME1" \
        --input2 "$INPUT2" --name2 "$NAME2" \
        --output "$OUTPUT_DIR" \
        --seed "$SEED" --k "$KVAL"
done

echo "All comparisons for $NAME1 finished."
deactivate
