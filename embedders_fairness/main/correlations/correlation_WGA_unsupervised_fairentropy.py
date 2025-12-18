import pandas as pd
import os
import numpy as np
import argparse
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr, kendalltau
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
parser.add_argument("--k", type=int, required=True)
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
# Step 2 : Load score (worst IS across clusters)
######################################################
print("Step 2: Load score")

import os
import pandas as pd
from collections import defaultdict

input_dir = f"__results/seed{seed}/K{k}/score"
size_dir = f"__results/seed{seed}/K{k}/clustering"

# Step 1: collect per-cluster medians
per_cluster_median = defaultdict(dict)

for filename in os.listdir(input_dir):
    if not filename.endswith(".csv") or not filename.startswith("cluster"): #cluster0_DistilRoBERTa.csv
        continue

    cluster_tag, model_x = filename.replace(".csv", "").split("__", 1) #cluster0, DistilRoBERTa



    score = pd.read_csv(os.path.join(input_dir, filename)).iloc[0, 1] #581
    per_cluster_median[cluster_tag].setdefault(model_x, []).append(float(score)) # {"cluster0": {"DistilRoBERTa": [581.0, ...]}, ...}


# take median per cluster
for cluster, models in per_cluster_median.items():
    for model_i, vals in models.items():
        per_cluster_median[cluster][model_i] = float(pd.Series(vals).median()) # {"cluster0": {"DistilRoBERTa": 581.0, "BERT": 602.5, ...}, "cluster1": {"DistilRoBERTa": 590.0, ...}, ...}


# Step 2: compute worst IS per embedder and track cluster
records = []
for model_i in embedder_names:
    cluster_scores = {c: per_cluster_median[c][model_i] for c in per_cluster_median if model_i in per_cluster_median[c]}  # per model: {"cluster0": 581.0, "cluster1": 590.0, ...}
    
    if not cluster_scores:
        print(f"[WARNING] No score found for embedder {model_i}, skipping.")
        continue

    worst_cluster, worst_val = min(cluster_scores.items(), key=lambda kv: kv[1])

    # Load cluster sizes for this embedder
    size_file = os.path.join(size_dir, f"{model_i}_cluster_size.csv")
    if os.path.exists(size_file):
        df_sizes = pd.read_csv(size_file)
        sizes = df_sizes["number_of_samples"].tolist()
        worst_cluster_idx = int(worst_cluster.replace("cluster", ""))
        worst_cluster_size = sizes[worst_cluster_idx]
        min_cluster_size = min(sizes)
        mean_cluster_size = int(sum(sizes) / len(sizes))
        max_cluster_size = max(sizes)
    else:
        worst_cluster_size = min_cluster_size = mean_cluster_size = max_cluster_size = None

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
print("\n🔍 Summary of worst IS per embedder:")
summary_cols = ["score_worst", "worst_cluster", "worst_cluster_size",
                "min_cluster_size", "mean_cluster_size", "max_cluster_size"]
print(df[summary_cols].dropna())




######################################################
# Step 3 : Worst accuracy per embedder (global model) MODIFIED FOR BETTER GOOD
######################################################
print("Step 3: Compute worst accuracy per embedder using a global model")

group_columns = ["SEX", "RAC1P"] #WARNING HARD DEFINED HERE
first_embedder = embedder_names[0]
df_ref = pd.read_csv(f"{args.data_path}/{first_embedder}.csv")
unique_values = {col: pd.unique(df_ref[col]) for col in group_columns}


embedding_dir = args.data_path
embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith(".csv")]



# Tags lisibles comme "SEXmanRAC1Pwhite"
import itertools
group_tags = {}
for idx, combo in enumerate(itertools.product(*[unique_values[col] for col in group_columns])):
    filters = dict(zip(group_columns, combo))
    tag = "".join(f"{col}{val}" for col, val in filters.items())
    group_tags[tag] = filters

# tailles de groupes (optionnel mais pratique)
cluster_sizes = []
for idx, (tag, filters) in enumerate(group_tags.items()):
    mask = pd.Series(True, index=df_ref.index)
    for col, val in filters.items():
        mask &= df_ref[col] == val
    cluster_sizes.append({"cluster_id": tag, "number_of_samples": int(mask.sum())})

shared_sizes = pd.DataFrame(cluster_sizes)
sizes = shared_sizes["number_of_samples"].tolist()

# Store results
records = []

for filename in embedding_files:
    filepath = os.path.join(embedding_dir, filename)
    model_name = filename.replace(".csv", "")

    df_embed = pd.read_csv(filepath)

    X_train_list, y_train_list = [], []
    test_sets = {}

    for group_name, filters in group_tags.items():
        mask = pd.Series(True, index=df_embed.index)
        for col, val in filters.items():
            mask &= df_embed[col] == val
        selected_ids = df_embed[mask].index

        if len(selected_ids) < 20:
            continue

        X = df_embed.iloc[selected_ids, :df_embed.columns.get_loc("target")].values
        y = df_embed.iloc[selected_ids]["target"].values if "target" in df_embed.columns else None
        if y is None:
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            test_sets[group_name] = (X_test, y_test)

            print(f"  {model_name} | {group_name} | Test samples: {len(X_test)}")

        except Exception as e:
            print(f"  Skipped {model_name} | {group_name} due to split error: {e}")

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

    X_train_global = np.vstack(X_train_list)
    y_train_global = np.concatenate(y_train_list)

    clf = LogisticRegression(solver="lbfgs", max_iter=2000)
    clf.fit(X_train_global, y_train_global)

    print(f"  Total training samples used: {len(y_train_global)}")

    cluster_accs = {}
    for group_name, (X_test, y_test) in test_sets.items():
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cluster_accs[group_name] = acc

        score = per_cluster_median.get(group_name, {}).get(model_name, None)
        print(f"  -> {model_name} | {group_name} | Accuracy: {acc:.4f}")

    print("\n")

    worst_cluster, worst_acc = min(cluster_accs.items(), key=lambda kv: kv[1])

    worst_cluster_idx = list(group_tags.keys()).index(worst_cluster)
    worst_cluster_size = sizes[worst_cluster_idx]
    min_cluster_size = min(sizes)
    mean_cluster_size = sum(sizes) / len(sizes)
    max_cluster_size = max(sizes)

    records.append({
        "embedder": model_name,
        "Accuracy_worst": worst_acc,
        "worst_cluster": worst_cluster,
        "worst_cluster_size": worst_cluster_size,
        "min_cluster_size": min_cluster_size,
        "mean_cluster_size": mean_cluster_size,
        "max_cluster_size": max_cluster_size,
    })

df_acc = pd.DataFrame(records).set_index("embedder").sort_index()
output_path = os.path.join(output_dir, "Accuracy_worst_details.csv")
df_acc.to_csv(output_path)
print(f"✅ Accuracy_worst_details saved to {output_path}")

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

# Print summay IS worst and Acc worst
summary_df = pd.DataFrame({
    "Accuracy_worst": accuracy_worst_series,
    "score_worst": score_worst_series
}).dropna()
print("\n🔍 Summary per model:")
print(summary_df)

# Save summay IS worst and Acc worst
summary_path = os.path.join(output_dir, "Worst_Accuracy_vs_score_summary.csv")
summary_df.to_csv(summary_path, index=True)
print(f"✅ Summary saved to {summary_path}")
