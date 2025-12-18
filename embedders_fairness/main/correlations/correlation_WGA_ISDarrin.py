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

input_dir = os.path.join(os.path.dirname(output_dir), "score")


# Load embedder list from json
with open("embedders.json", "r") as f:
    embedder_names = list(json.load(f).keys())


# Dictionary: { U : [IS(U→V1), IS(U→V2), ...] }
score_per_embedder = defaultdict(list)

for filename in os.listdir(input_dir):
    if not filename.endswith(".csv"):
        continue

    # Example: DistilRoBERTa__VS__average_word_embeddings_glove.csv
    models_part = filename.replace(".csv", "")
    model_x, model_y = models_part.split("_VS_", 1)

    # Exclude self-comparisons (U == V)
    if model_x == model_y:
        continue

    # Each CSV contains a 1x1 pivot with the scorecore
    score = pd.read_csv(os.path.join(input_dir, filename)).iloc[0, 1]
    score_per_embedder[model_x].append(float(score))


######################################################
# Step 2 b : Compute median scoreper embedder
######################################################
records = []

for model_i in embedder_names:
    if model_i not in score_per_embedder:
        continue

    vals = score_per_embedder[model_i]
    median_val = float(pd.Series(vals).median())

    records.append({
        "embedder": model_i,
        "score_worst": median_val,
        "num_pairs": len(vals)
    })


######################################################
# Step : Save results
######################################################
df = pd.DataFrame(records).set_index("embedder").sort_index()
output_path = os.path.join(output_dir, "score_worst_details.csv")
df.to_csv(output_path)
print(f"✅ score_worst_details saved to {output_path}")

# Save only the median column
score_worst_df = df[["score_worst"]]
simple_path = os.path.join(output_dir, "score_worst.csv")
score_worst_df.to_csv(simple_path)
print(f"✅ score_worst saved to {simple_path}")


######################################################
# Step : Summary printout
######################################################
print("\n🔍 Summary of median scoreper embedder:")
summary_cols = ["score_worst", "num_pairs"]
print(df[summary_cols].dropna())







######################################################
# Step 2 : Train classifier and compute accuracy
######################################################
print("Step 2: Compute accuracy per embedder (global model)")

embedding_dir = args.data_path
embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith(".csv")]

records = []

for filename in embedding_files:
    filepath = os.path.join(embedding_dir, filename)
    model_name = filename.replace(".csv", "")

    # Load embeddings
    df_embed = pd.read_csv(filepath)

    if "target" not in df_embed.columns:
        print(f"⚠️ Skipping {model_name}: no 'target' column found")
        continue

    # Features and labels
    X = df_embed.iloc[:, :df_embed.columns.get_loc("target")].values
    y = df_embed["target"].values

    # Stratified train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train global classifier (logistic regression with standardization)
        clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42))
        #clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42) #ATTENTION TO CHANGE

        clf.fit(X_train, y_train)

        # Evaluate accuracy on test set
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"  -> {model_name} | Accuracy: {acc:.4f}")

        records.append({
            "embedder": model_name,
            "Accuracy_worst": acc,
            "num_train_samples": len(y_train),
            "num_test_samples": len(y_test),
        })

    except Exception as e:
        print(f"⚠️ Skipped {model_name} due to error: {e}")
        records.append({
            "embedder": model_name,
            "Accuracy_worst": None,
            "num_train_samples": None,
            "num_test_samples": None,
        })


######################################################
# Step 3 : Save results
######################################################
df_acc = pd.DataFrame(records).set_index("embedder").sort_index()
output_path = os.path.join(output_dir, "Accuracy_worst_details.csv")
df_acc.to_csv(output_path)
print(f"✅ Accuracy_worst_details saved to {output_path}")

# Save only the accuracy column
accuracy_worst_series = df_acc[["Accuracy_worst"]].dropna()
accuracy_worst_path = os.path.join(output_dir, "Accuracy_worst.csv")
accuracy_worst_series.to_csv(accuracy_worst_path, header=True)
print(f"✅ Accuracy_worst saved to {accuracy_worst_path}")


######################################################
# Step 4 : Summary printout
######################################################
print("\n🔍 Summary of accuracy per embedder:")
summary_cols = ["Accuracy_worst", "num_train_samples", "num_test_samples"]
print(df_acc[summary_cols].dropna())






######################################################
# Step 4 : Correlation analysis
######################################################
print("Step 4: Correlation analysis")

# Load accuracy and scoremedian results
accuracy_worst_series = pd.read_csv(
    os.path.join(output_dir, "Accuracy_worst.csv"),
    index_col=0
).squeeze("columns")

score_worst_series = pd.read_csv(
    os.path.join(output_dir, "score_worst.csv"),
    index_col=0
).squeeze("columns")

# Keep only common embedders
common_models = accuracy_worst_series.index.intersection(score_worst_series.index)
x = accuracy_worst_series.loc[common_models].values
y = score_worst_series.loc[common_models].values

# Define correlation methods
methods = {
    "Pearson": pearsonr,
    "Spearman": spearmanr,
    "Kendall": kendalltau
}

# Compute and print correlation results
corr_results = {}
for name, func in methods.items():
    corr, pval = func(x, y)
    corr_results[name] = {"corr": corr, "pval": pval}
    print(f"{name:<10}: corr = {corr:.4f}, p = {pval:.4g}")

# Save correlation results to CSV
corr_df = pd.DataFrame.from_dict(corr_results, orient="index")
corr_df.index.name = "method"
corr_path = os.path.join(output_dir, "Worst_Accuracy_vs_score_correlation.csv")
corr_df.to_csv(corr_path, index=True)
print(f"✅ Correlation results saved to {corr_path}")

# Build summary of Accuracy and scorecores per embedder
summary_df = pd.DataFrame({
    "Accuracy_worst": accuracy_worst_series,
    "score_worst": score_worst_series
}).dropna()

print("\n🔍 Summary per model:")
print(summary_df)

# Save summary to CSV
summary_path = os.path.join(output_dir, "Worst_Accuracy_vs_score_summary.csv")
summary_df.to_csv(summary_path, index=True)
print(f"✅ Summary saved to {summary_path}")