import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau




# ----- WARNING: START OF THE MODIFYING SECTION BASED ON WHAT TO PLOT -----
# Load the saved data
path = "__results/seed1/KSEX_RAC1P/correlations"
#path = "__results/seed1/K2/correlations"
# ----- WARNING: END OF THE MODIFYING SECTION -----




accuracy_df = pd.read_csv(f"{path}/Accuracy_worst.csv", index_col=0).squeeze("columns")
is_df = pd.read_csv(f"{path}/IS_score_worst.csv", index_col=0).squeeze("columns")

# Merge the two to keep only the common models
common_models = accuracy_df.index.intersection(is_df.index)
df_corr = pd.DataFrame({
    "Accuracy_worst": accuracy_df.loc[common_models],
    "IS_score_worst": is_df.loc[common_models]
})

# WARNING: If needed drop "SFR-Embedding-Mistral"
df_corr = df_corr.drop(index="SFR-Embedding-Mistral", errors="ignore")

# Simple scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_corr, x="IS_score_worst", y="Accuracy_worst", s=100)

# Annotate the points
for model, row in df_corr.iterrows():
    plt.text(row["IS_score_worst"], row["Accuracy_worst"], model, fontsize=9)

plt.title("IS_score_worst vs Accuracy_worst")
plt.xlabel("IS_score_worst")
plt.ylabel("Accuracy_worst")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{path}/IS_vs_Accuracy_scatter_no_size.png")
plt.show()

# Calculate and display the correlations
x, y = df_corr["IS_score_worst"], df_corr["Accuracy_worst"]

methods = {
    "Pearson": pearsonr,
    "Spearman": spearmanr,
    "Kendall": kendalltau
}

corr_results = {}
for name, func in methods.items():
    corr, pval = func(x, y)
    corr_results[name] = {"corr": corr, "pval": pval}
    print(f"{name:<10}: corr = {corr:.4f}, p = {pval:.4g}")
