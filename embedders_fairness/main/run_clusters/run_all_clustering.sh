#!/bin/bash
#SBATCH --account=def-umaivodj


# Arguments
OUTPUT_DIR="$1"   # e.g. __results/seed_1/K2/clustering

# Path to JSON file
EMBEDDERS_JSON="./embedders.json"

count=0
while read -r name path; do

    echo "Submitting clustering job: $name (seed=$SEED, K=$KVAL)"
    csv_path="${DATA_PATH}/${name}.csv"

    sbatch \
    --job-name="clustering_${name}_s${SEED}_k${KVAL}" \
    --output="${OUTPUT_DIR}/logs/${name}_%j.out" \
    --error="${OUTPUT_DIR}/logs/${name}_%j.err" \
    run_clusters/run_single_clustering.sh "$csv_path" "$OUTPUT_DIR" "$name" "$SEED" "$KVAL"

    ((count++))
done < <(python -c "import json; d=json.load(open('$EMBEDDERS_JSON')); [print(k, v) for k,v in d.items()]")

echo "Total clustering jobs submitted: $count"
