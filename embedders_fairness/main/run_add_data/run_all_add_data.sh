#!/bin/bash
#SBATCH --account=def-umaivodj
#SBATCH --job-name=run_all_add_data
#SBATCH --output=./__logs/run_all_add_data_%j.out
#SBATCH --error=./__logs/run_all_add_data_%j.err


# Path to JSON file
EMBEDDERS_JSON="./embedders.json"

count=0
while read -r name path; do
    INPUT="${DATA_PATH}/${name}.csv"

    echo "Submitting adding data job: $name"
    sbatch \
        --job-name="adding_data_${name}" \
        --output="./__results/data/logs_add_data/${name}_%j.out" \
        --error="./__results/data/logs_add_data/${name}_%j.err" \
        run_add_data/run_single_add_data.sh "$INPUT" "$name" "$RAW_DATA_PATH"
    ((count++))
done < <(python -c "import json; d=json.load(open('$EMBEDDERS_JSON')); [print(k, v) for k,v in d.items()]")

echo "Total adding data jobs submitted: $count"
