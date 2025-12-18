#!/bin/bash
#SBATCH --account=def-umaivodj
#SBATCH --job-name=run_all_clustering
#SBATCH --output=./__logs/run_all_clustering_%j.out
#SBATCH --error=./__logs/run_all_clustering_%j.err



# NORMAL CODE WHEN RUN ALL STEPS
# Path to JSON file
EMBEDDERS_JSON="./embedders.json"

# Loop through JSON entries (key = name, value = path)
count=0
while read -r name path; do
    output_file="${DATA_PATH}/${name}.csv"   # <-- adapte si ce n’est pas le bon format

    if [ -f "$output_file" ]; then
        echo "⏩ Embeddings for $name already exist ($output_file), skipping."
        continue
    fi

    echo "Submitting embedding job: $name"
    sbatch \
    --job-name="embed_${name}" \
    --output=./__results/data/logs_embeddings/${name}_%j.out \
    --error=./__results/data/logs_embeddings/${name}_%j.err \
    run_embeddings/run_single_embedding_text.sh "$name" "$path" "$DATA_PATH" "$RAW_DATA_PATH"

    ((count++))
done < <(python -c "import json; d=json.load(open('$EMBEDDERS_JSON')); [print(k, v) for k,v in d.items()]")

echo "Total jobs submitted: $count"
echo "All embedding jobs have been submitted."
