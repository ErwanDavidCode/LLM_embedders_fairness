#!/bin/bash
#SBATCH --account=def-umaivodj

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G #32G


echo "Job started on $(hostname)"

module load python/3.11.5
source ../../new_venv/bin/activate
echo "Running correlation_WGA_ISDarrin.py..."

# Variables from pipeline
OUTPUT_DIR="$1"

python correlations/correlation_WGA_ISDarrin.py \
    --output "$OUTPUT_DIR" \
    --data_path "$DATA_PATH" \
    --seed "$SEED" --k "$KVAL"

deactivate
echo "Job completed."