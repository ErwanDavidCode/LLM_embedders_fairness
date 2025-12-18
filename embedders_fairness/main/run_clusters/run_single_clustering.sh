#!/bin/bash
#SBATCH --account=def-umaivodj
#SBATCH --job-name=run_single_clustering

#SBATCH --time=00:45:00 #Good for 100_000 samples from ACS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "Running on $(hostname)"

# Load Python environment
module load python/3.11.5
source ../../new_venv/bin/activate

# Arguments
csv_path="$1"    # embeddings dataset
OUTPUT_DIR="$2"   # output CSV with clusters
name="$3"
SEED="$4"
KVAL="$5"

echo "Launching clustering for $csv_path with K=$KVAL, seed=$SEED"
python run_clusters/clustering.py --csv_path "$csv_path" --output "$OUTPUT_DIR" --name "$name" --seed "$SEED" --k "$KVAL"

deactivate
echo "Job finished."
