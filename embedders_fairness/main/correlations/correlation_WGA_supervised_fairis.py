import pandas as pd
import os
import numpy as np
import argparse
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import itertools
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Force Pandas to display all rows and columns (no truncation)
pd.set_option("display.max_rows", None)        # Show all rows
pd.set_option("display.max_columns", None)     # Show all columns
pd.set_option("display.width", None)           # Auto-detect console width
pd.set_option("display.max_colwidth", None)    # Do not truncate column content



######################################################
# Step 1 : Parse arguments
######################################################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--k", type=str, required=True)
parser.add_argument("--output", type=str, required=True, help="Output directory for correlation results")
parser.add_argument("--data_path", type=str, required=True, help="Input directory for the embeddings")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

seed = args.seed
k = args.k
output_dir = args.output



# Load embedder list from json
with open("embedders.json", "r") as f:
    embedder_names = list(json.load(f).keys())


######################################################
# Step 2 : Load score (worst scoreacross clusters)
######################################################
print("Step 2: Load score")


input_dir = os.path.join(os.path.dirname(output_dir), "score")
size_dir = os.path.join(os.path.dirname(output_dir), "groups_with_demog")


# Step 1: collect per-cluster medians
per_cluster_median = defaultdict(dict)

for filename in os.listdir(input_dir):
    if not filename.endswith(".csv"):
        continue

    # Example: SEXmanRAC1Pnon_white__DistilRoBERTa_VS_average_word_embeddings_glove.csv
    cluster_tag, models_part = filename.replace(".csv", "").split("__", 1)

    model_x, model_y = models_part.split("_VS_", 1)






    # new (U != V)
    if model_x == model_y:
        continue  # exclure l’auto-paire
    
    
    
    

    
    
    
    score = pd.read_csv(os.path.join(input_dir, filename)).iloc[0, 1]  # 1x1 pivot
    per_cluster_median[cluster_tag].setdefault(model_x, []).append(float(score))


# Take median per cluster per embedder
for cluster, models in per_cluster_median.items():
    for model_i, vals in models.items():
        per_cluster_median[cluster][model_i] = float(pd.Series(vals).median())


######################################################
# Step 2b : Compute group proportions once (for demog mode)
######################################################
group_columns = k.split("_")
first_embedder = embedder_names[0]
df_ref = pd.read_csv(f"{args.data_path}/{first_embedder}.csv")

unique_values = {col: pd.unique(df_ref[col]) for col in group_columns}
cluster_sizes = []
for idx, combo in enumerate(itertools.product(*[unique_values[col] for col in group_columns])):
    filters = dict(zip(group_columns, combo))
    mask = pd.Series(True, index=df_ref.index)
    for col, val in filters.items():
        mask &= df_ref[col] == val
    cluster_name = "".join(f"{col}{val}" for col, val in filters.items())
    cluster_sizes.append({"cluster_id": cluster_name, "number_of_samples": int(mask.sum())})

# Save a single shared file
os.makedirs(size_dir, exist_ok=True)
size_file = os.path.join(size_dir, "group_cluster_size.csv")
pd.DataFrame(cluster_sizes).to_csv(size_file, index=False)

shared_sizes = pd.read_csv(size_file)
sizes = shared_sizes["number_of_samples"].tolist()


######################################################
# Step 3 : Compute worst scoreper embedder
######################################################
records = []

for model_i in embedder_names:
    cluster_score = {
        c: per_cluster_median[c][model_i]
        for c in per_cluster_median if model_i in per_cluster_median[c]
    }
    worst_cluster, worst_val = min(cluster_score.items(), key=lambda kv: kv[1])



    # # Map cluster tag to its index position
    # worst_cluster_idx = list(per_cluster_median.keys()).index(worst_cluster)
    # worst_cluster_size = sizes[worst_cluster_idx]

    size_map = dict(zip(shared_sizes["cluster_id"], shared_sizes["number_of_samples"]))
    worst_cluster_size = size_map[worst_cluster]



    min_cluster_size = min(sizes)
    mean_cluster_size = int(sum(sizes) / len(sizes))
    max_cluster_size = max(sizes)

    # Record results
    records.append({
        "embedder": model_i,
        "score_worst": worst_val,
        "worst_cluster": worst_cluster,
        "worst_cluster_size": worst_cluster_size,
        "min_cluster_size": min_cluster_size,
        "mean_cluster_size": mean_cluster_size,
        "max_cluster_size": max_cluster_size,
    })



# Save full details
df = pd.DataFrame(records).set_index("embedder").sort_index()
output_path = os.path.join(output_dir, "score_worst_details.csv")
df.to_csv(output_path)
print(f"✅ score_worst_details saved to {output_path}")

# Save score worst
score_worst_df = df[["score_worst"]]
simple_path = os.path.join(output_dir, "score_worst.csv")
score_worst_df.to_csv(simple_path)
print(f"✅ score_worst saved to {simple_path}")



######################################################
# Step 2b : Summary printout
######################################################
print("\n🔍 Summary of worst scoreper embedder:")
summary_cols = ["score_worst", "worst_cluster", "worst_cluster_size",
                "min_cluster_size", "mean_cluster_size", "max_cluster_size"]
print(df[summary_cols].dropna())




######################################################
# Step 3 : Worst accuracy per embedder (global model)
######################################################
print("Step 3: Compute worst accuracy per embedder using a global model")

embedding_dir = args.data_path

# embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith(".csv")]
import json
with open("embedders.json", "r") as f:
    embedder_names = json.load(f).keys()
embedding_files = [f"{name}.csv" for name in embedder_names]



# Reuse the same group_columns and unique_values from Step 2b
group_tags = {}
for idx, combo in enumerate(itertools.product(*[unique_values[col] for col in group_columns])):
    filters = dict(zip(group_columns, combo))
    # Build a readable tag (same as Step 2b ordering)
    tag = "".join(f"{col}{val}" for col, val in filters.items())
    group_tags[tag] = filters




records = []

for filename in embedding_files:
    filepath = os.path.join(embedding_dir, filename) #"__results/data", "DistilRoBERTa-paraphrase.csv"
    model_name = filename.replace(".csv", "")

    # Load embeddings
    df_embed = pd.read_csv(filepath)

    # Dictionaries to store train/test splits per cluster
    X_train_list, y_train_list = [], []
    test_sets = {}

    for cluster_name, cluster_id in group_tags.items():
        filters = cluster_id  # here cluster_id is actually the dict of {col: value}
        mask = pd.Series(True, index=df_embed.index)
        for col, val in filters.items():
            mask &= df_embed[col] == val
        selected_ids = df_embed[mask].index


        if len(selected_ids) < 20:
            continue

        # Features and labels
        X = df_embed.iloc[selected_ids, :df_embed.columns.get_loc("target")].values
        y = df_embed.iloc[selected_ids]["target"].values if "target" in df_embed.columns else None

        
        if y is None:
            continue

        # Stratified split par groupe
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y) #stratify : le split s'arrage pour avoir la meme proportions de classes y en train et test
            X_train_list.append(X_train)
            y_train_list.append(y_train)

            test_sets[cluster_name] = (X_test, y_test)

            # --- Size of test ---
            print(f"  {model_name} | {cluster_name} | Test samples: {len(X_test)}")

        except Exception as e:
            print(f"  Skipped {model_name} | {cluster_name} due to split error: {e}")

    if not X_train_list:
        records.append({
            "embedder": model_name,
            "Accuracy_worst": None,
            "worst_cluster": None,
            "worst_cluster_size": None,
            "min_cluster_size": None,
            "mean_cluster_size": None,
            "max_cluster_size": None,
        })
        continue
 
    # Train global model
    X_train_global = np.vstack(X_train_list)
    y_train_global = np.concatenate(y_train_list)

    clf = LogisticRegression(solver="lbfgs", max_iter=2000)
    #clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42))
    clf.fit(X_train_global, y_train_global)
    
    # --- Size of test ---
    print(f"  Total training samples used: {len(y_train_global)}")

    # Compute accuracy per cluster on its test set
    cluster_accs = {}
    for cluster_name, (X_test, y_test) in test_sets.items():
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cluster_accs[cluster_name] = acc

        # Optional: score if available
        score = per_cluster_median.get(cluster_name, {}).get(model_name, None)
        print(f"  -> {model_name} | {cluster_name} | Accuracy: {acc:.4f}" + (f" | IS: {score:.4f}" if score else " | IS: N/A"))

    print("\n")
    # Identify worst cluster
    worst_cluster, worst_acc = min(cluster_accs.items(), key=lambda kv: kv[1])

    # Load cluster sizes
    df_sizes = shared_sizes  # already loaded in Step 2b
    sizes = df_sizes["number_of_samples"].tolist()
    worst_cluster_idx = list(group_tags.keys()).index(worst_cluster)

    worst_cluster_size = sizes[worst_cluster_idx]
    min_cluster_size = min(sizes)
    mean_cluster_size = sum(sizes) / len(sizes)
    max_cluster_size = max(sizes)


    # Add record
    records.append({
        "embedder": model_name,
        "Accuracy_worst": worst_acc,
        "worst_cluster": worst_cluster,
        "worst_cluster_size": worst_cluster_size,
        "min_cluster_size": min_cluster_size,
        "mean_cluster_size": mean_cluster_size,
        "max_cluster_size": max_cluster_size,
    })

# Build DataFrame and save
df_acc = pd.DataFrame(records).set_index("embedder").sort_index()
output_path = os.path.join(output_dir, "Accuracy_worst_details.csv")
df_acc.to_csv(output_path)
print(f"✅ Accuracy_worst_details saved to {output_path}")

# Save only the worst accuracy values
accuracy_worst_series = df_acc[["Accuracy_worst"]].dropna()
accuracy_worst_path = os.path.join(output_dir, "Accuracy_worst.csv")
accuracy_worst_series.to_csv(accuracy_worst_path, header=True)
print(f"✅ Accuracy_worst saved to {accuracy_worst_path}")



######################################################
# Step 3b : Summary printout
######################################################
print("\n🔍 Summary of worst Accuracy per embedder:")
summary_cols = [
    "Accuracy_worst", "worst_cluster", "worst_cluster_size",
    "min_cluster_size", "mean_cluster_size", "max_cluster_size"
]
print(df_acc[summary_cols].dropna())


######################################################
# Step 4 : Correlation analysis
######################################################
print("Step 4: Correlation analysis")

# load csv from above
accuracy_worst_series = pd.read_csv(
    os.path.join(output_dir, "Accuracy_worst.csv"),
    index_col=0
).squeeze("columns")
score_worst_series = pd.read_csv(
    os.path.join(output_dir, "score_worst.csv"),
    index_col=0
).squeeze("columns")

common_models = accuracy_worst_series.index.intersection(score_worst_series.index)
x = accuracy_worst_series.loc[common_models].values
y = score_worst_series.loc[common_models].values

methods = {
    "Pearson": pearsonr,
    "Spearman": spearmanr,
    "Kendall": kendalltau
}

# Print correlation results
corr_results = {}
for name, func in methods.items():
    corr, pval = func(x, y)
    corr_results[name] = {"corr": corr, "pval": pval}
    print(f"{name:<10}: corr = {corr:.4f}, p = {pval:.4g}")


# Save correlation results
corr_df = pd.DataFrame.from_dict(corr_results, orient="index")
corr_df.index.name = "method"
corr_path = os.path.join(output_dir, "Worst_Accuracy_vs_score_correlation.csv")
corr_df.to_csv(corr_path, index=True)
print(f"✅ Correlation results saved to {corr_path}")

# Print summay score worst and Acc worst
summary_df = pd.DataFrame({
    "Accuracy_worst": accuracy_worst_series,
    "score_worst": score_worst_series
}).dropna()
print("\n🔍 Summary per model:")
print(summary_df)

# Save summay score worst and Acc worst
summary_path = os.path.join(output_dir, "Worst_Accuracy_vs_score_summary.csv")
summary_df.to_csv(summary_path, index=True)
print(f"✅ Summary saved to {summary_path}")
