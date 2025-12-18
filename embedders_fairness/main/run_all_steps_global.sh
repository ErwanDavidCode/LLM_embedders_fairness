#!/bin/bash
#SBATCH --account=def-umaivodj
#SBATCH --job-name=run_pipeline
#SBATCH --output=./__results/pipeline_%j.out
#SBATCH --error=./__results/pipeline_%j.err
#SBATCH --time=02:00:00 #6:00:00 #entire pipeline - 100 000 data


# Start time
echo "⏱️ Pipeline started at $(date)"
start_time=$(date +%s)

# Global setup
mkdir -p logs
mkdir -p completion_flags
mkdir -p __results

# Global variables
NUM_EMBEDDERS=$(python -c "import json; print(len(json.load(open('embedders.json'))))")
NUM_PAIRS=$((NUM_EMBEDDERS * NUM_EMBEDDERS))
echo "NUM_EMBEDDERS = ${NUM_EMBEDDERS} and NUM_PAIRS = ${NUM_PAIRS}"

# WARNING: to be modified considering the dataset
DATASET="ACSEmployment"
RAW_DATA_PATH="../../Data/datasets/100_000/${DATASET}_with_target.csv" #Useful for add_data & embeddings only
DATA_PATH="../../Data/embeddings/100_000/${DATASET}" #The folder has to exist (even if empty), is filled by step 1 embeddings and step 1 add data
echo "My data path is: $DATA_PATH"

# SCORE used: FAIR_IS or FAIR_ENTROPY
SCORE="FAIR_ENTROPY"
echo "My score is: $SCORE"
if [ "$SCORE" = "FAIR_ENTROPY" ]; then
    STOP_CONDITION="$NUM_EMBEDDERS"
elif [ "$SCORE" = "FAIR_IS" ]; then
    STOP_CONDITION="$NUM_PAIRS"
else
    echo "Unknown score: $SCORE"
    exit 1
fi



# -----------------------
# Step 1: Embeddings - 
# WARNING: the dictionary 'PUMS_Data_Dictionary_2018.csv' file has to be specified in the .py
# WARNING: if it has been run for less than 'NUM_EMBEDDERS' numbers of embedders, it will not pass the flag test and won't continue the pipeline. It would have to be stopped and re run manually after this step.
# -----------------------
echo "[STEP 1] Starting embeddings..."
mkdir -p completion_flags/step1_embeddings
rm -f completion_flags/step1_embeddings/*.done
mkdir -p ./__results/data/logs_embeddings

DATA_PATH=$DATA_PATH RAW_DATA_PATH=$RAW_DATA_PATH sbatch \
    --job-name="ALL_embeddings" \
    --output=./__results/data/logs_embeddings/ALL_embeddings_%j.out \
    --error=./__results/data/logs_embeddings/ALL_embeddings_%j.err \
    run_embeddings/run_all_embeddings.sh # or just run_all_embeddings_text.sh if tabular data

echo "[STEP 1] Waiting for $NUM_EMBEDDERS flags (checking every 5 min)..."
while [ "$(ls completion_flags/step1_embeddings/*.done 2>/dev/null | wc -l)" -lt "$NUM_EMBEDDERS" ]; do
    echo "[STEP 1] $(date) — Current flags: $(ls completion_flags/step1_embeddings/*.done 2>/dev/null | wc -l)/$NUM_EMBEDDERS"
    sleep 300
done
echo "[STEP 1] ✅ All jobs completed."



 
# # -----------------------
# # Step 1 (OPTIONAL): Add data about sensitive attributes (necessary if the embeddings dataset per embedder don't have the needed columns at the end)
# # -----------------------
# echo "[STEP 1] Adding data..."
# mkdir -p completion_flags/step1_add_data
# rm -f completion_flags/step1_add_data/*.done
# mkdir -p ./__results/data/logs_add_data

# DATA_PATH=$DATA_PATH RAW_DATA_PATH=$RAW_DATA_PATH sbatch \
#     --job-name="ALL_add_data" \
#     --output="./__results/data/logs_add_data/ALL_add_data%j.out" \
#     --error="./__results/data/logs_add_data/ALL_add_data%j.err" \
#     run_add_data/run_all_add_data.sh

# echo "[STEP 1] Waiting for $NUM_EMBEDDERS flags..."
# while [ "$(ls completion_flags/step1_add_data/*.done 2>/dev/null | wc -l)" -lt "$NUM_EMBEDDERS" ]; do
#     echo "[STEP 1] $(date) — Current flags: $(ls completion_flags/step1_add_data/*.done 2>/dev/null | wc -l)/$NUM_EMBEDDERS"
#     sleep 300
# done
# echo "[STEP 1] ✅ Adding data completed."



# -----------------------
# Step 2–4: Loop seeds × K
# -----------------------
seeds=(1) # EX: (1 2 3 4 5) or (1) for simple test
K_values=("SEX_RAC1P") # EX: (2 4 8) or ("SEX_RAC1P" "SEX") or ("ISDarrin"). Either a number that is the number of clusters (1, 2, 4, 16, 32 ...), or names of the sensitive attribute to consider separated by '_' in SEX, RAC1P or AGEP or "ISDarrin" for a normal IS score application

for seed in "${seeds[@]}"; do
    for K in "${K_values[@]}"; do
        echo "=== 🚀 Experiment seed=${seed}, K=${K} ==="

        # Folder setup
        res_dir="__results/seed${seed}/K${K}"
        mkdir -p "${res_dir}/clustering"
        mkdir -p "${res_dir}/add_data"
        mkdir -p "${res_dir}/groups_with_demog"
        mkdir -p "${res_dir}/score"
        mkdir -p "${res_dir}/score/loss_marg"
        mkdir -p "${res_dir}/score/loss_cond"
        mkdir -p "${res_dir}/correlations"


        # -----------------------
        # Step 2: Clustering
        # -----------------------
        if [[ "$K" =~ ^[0-9]+$ ]]; then
            OUTPUT_DIR="${res_dir}/clustering"
            echo "[STEP 2] Clustering (seed=${seed}, K=${K})"
            mkdir -p completion_flags/step2_clustering
            rm -f completion_flags/step2_clustering/*.done
            mkdir -p "${OUTPUT_DIR}/logs"

            SEED=$seed KVAL=$K DATA_PATH=$DATA_PATH sbatch \
            --job-name="ALL_clustering_s${SEED}_k${KVAL}" \
            --output="${OUTPUT_DIR}/logs/ALL_clustering_s${SEED}_k${KVAL}_%j.out" \
            --error="${OUTPUT_DIR}/logs/ALL_clustering_s${SEED}_k${KVAL}_%j.err" \
            run_clusters/run_all_clustering.sh "${OUTPUT_DIR}"

            echo "[STEP 2] Waiting for $NUM_EMBEDDERS flags..."
            while [ "$(ls completion_flags/step2_clustering/*.done 2>/dev/null | wc -l)" -lt "$NUM_EMBEDDERS" ]; do
                echo "[STEP 2] $(date) — Current flags: $(ls completion_flags/step2_clustering/*.done 2>/dev/null | wc -l)/$NUM_EMBEDDERS"
                sleep 120
            done
            echo "[STEP 2] ✅ Clustering completed."
        fi


        # -----------------------
        # Step 3: Score
        # -----------------------
        OUTPUT_DIR="${res_dir}/score"
        echo "[STEP 3] Score (seed=${seed}, K=${K})"
        mkdir -p completion_flags/step3_score
        rm -f completion_flags/step3_score/*.done
        mkdir -p "${OUTPUT_DIR}/logs"

        # If we don't know the sensitive attribute
        if [[ "$K" =~ ^[0-9]+$ ]]; then
            SEED=$seed KVAL=$K DATA_PATH=$DATA_PATH sbatch \
            --array=0-$(($NUM_EMBEDDERS-1)) \
            --job-name="ALL_score_s${SEED}_k${KVAL}" \
            --output="${OUTPUT_DIR}/logs/ALL_score_s${SEED}_k${KVAL}_%A_%a.out" \
            --error="${OUTPUT_DIR}/logs/ALL_score_s${SEED}_k${KVAL}_%A_%a.err" \
            emir/run_score/run_all_score_unsupervised_fairentropy.sh "${OUTPUT_DIR}"
        # Elif K == "ISDarrin"
        elif [[ "$K" == "ISDarrin" ]]; then
            SEED=$seed KVAL=$K DATA_PATH=$DATA_PATH sbatch \
            --array=0-$(($NUM_EMBEDDERS-1)) \
            --job-name="ALL_score_s${SEED}_k${KVAL}" \
            --output="${OUTPUT_DIR}/logs/ALL_score_s${SEED}_k${KVAL}_%A_%a.out" \
            --error="${OUTPUT_DIR}/logs/ALL_score_s${SEED}_k${KVAL}_%A_%a.err" \
            emir/run_score/run_all_score_ISDarrin.sh "${OUTPUT_DIR}"       
        # If we know the sensitive attribute
        else
            if [ "$SCORE" = "FAIR_ENTROPY" ]; then
                SEED=$seed KVAL=$K DATA_PATH=$DATA_PATH sbatch \
                --array=0-$(($NUM_EMBEDDERS-1)) \
                --job-name="ALL_score_s${SEED}_k${KVAL}" \
                --output="${OUTPUT_DIR}/logs/ALL_score_s${SEED}_k${KVAL}_%A_%a.out" \
                --error="${OUTPUT_DIR}/logs/ALL_score_s${SEED}_k${KVAL}_%A_%a.err" \
                emir/run_score/run_all_score_supervised_fairentropy.sh "${OUTPUT_DIR}"
            elif [ "$SCORE" = "FAIR_IS" ]; then
                SEED=$seed KVAL=$K DATA_PATH=$DATA_PATH sbatch \
                --array=0-$(($NUM_EMBEDDERS-1)) \
                --job-name="ALL_score_s${SEED}_k${KVAL}" \
                --output="${OUTPUT_DIR}/logs/ALL_score_s${SEED}_k${KVAL}_%A_%a.out" \
                --error="${OUTPUT_DIR}/logs/ALL_score_s${SEED}_k${KVAL}_%A_%a.err" \
                emir/run_score/run_all_score_supervised_fairis.sh "${OUTPUT_DIR}"
            else
                echo "Unknown SCORE: $SCORE"
                exit 1
            fi
        fi

        echo "[STEP 3] Waiting for $STOP_CONDITION flags (SCORE used is $SCORE)..."
        while [ "$(ls completion_flags/step3_score/*.done 2>/dev/null | wc -l)" -lt "$STOP_CONDITION" ]; do
            echo "[STEP 3] $(date) — Current flags: $(ls completion_flags/step3_score/*.done 2>/dev/null | wc -l)/$STOP_CONDITION"
            sleep 120
        done
        echo "[STEP 3] ✅ Scores completed."



        # -----------------------
        # Step 4: Correlations
        # -----------------------
        OUTPUT_DIR="${res_dir}/correlations"
        echo "[STEP 4] Correlations (seed=${seed}, K=${K})"
        mkdir -p "${OUTPUT_DIR}/logs"

        # If we don't know the sensitive attribute
        if [[ "$K" =~ ^[0-9]+$ ]]; then
            SEED=$seed KVAL=$K DATA_PATH=$DATA_PATH sbatch \
            --job-name="correlations_s${SEED}_k${KVAL}" \
            --output="${OUTPUT_DIR}/logs/correlations_s${SEED}_k${KVAL}_%A_%a.out" \
            --error="${OUTPUT_DIR}/logs/correlations_s${SEED}_k${KVAL}_%A_%a.err" \
            correlations/run_correlation_different_k.sh "${OUTPUT_DIR}"
        # Elif K == "ISDarrin"
        elif [[ "$K" == "ISDarrin" ]]; then
            SEED=$seed KVAL=$K DATA_PATH=$DATA_PATH sbatch \
            --job-name="correlations_s${SEED}_k${KVAL}" \
            --output="${OUTPUT_DIR}/logs/correlations_s${SEED}_k${KVAL}_%A_%a.out" \
            --error="${OUTPUT_DIR}/logs/correlations_s${SEED}_k${KVAL}_%A_%a.err" \
            correlations/run_correlation_ISDarrin.sh "${OUTPUT_DIR}"
        # If we know the sensitive attribute
        else
            if [ "$SCORE" = "FAIR_ENTROPY" ]; then
                SEED=$seed KVAL=$K DATA_PATH=$DATA_PATH sbatch \
                --job-name="correlations_s${SEED}_k${KVAL}" \
                --output="${OUTPUT_DIR}/logs/correlations_s${SEED}_k${KVAL}_%A_%a.out" \
                --error="${OUTPUT_DIR}/logs/correlations_s${SEED}_k${KVAL}_%A_%a.err" \
                correlations/run_correlation_with_demog_entropy.sh "${OUTPUT_DIR}"
            elif [ "$SCORE" = "FAIR_IS" ]; then
                SEED=$seed KVAL=$K DATA_PATH=$DATA_PATH sbatch \
                --job-name="correlations_s${SEED}_k${KVAL}" \
                --output="${OUTPUT_DIR}/logs/correlations_s${SEED}_k${KVAL}_%A_%a.out" \
                --error="${OUTPUT_DIR}/logs/correlations_s${SEED}_k${KVAL}_%A_%a.err" \
                correlations/run_correlation_with_demog.sh "${OUTPUT_DIR}"
            else
                echo "Unknown SCORE: $SCORE"
                exit 1
            fi
        fi

    done
done

# You can execute final_plot.py after all seeds/K have been run to get the final plots for every seed and K
# For final_plot.py, don't forget to modify the file path inside it