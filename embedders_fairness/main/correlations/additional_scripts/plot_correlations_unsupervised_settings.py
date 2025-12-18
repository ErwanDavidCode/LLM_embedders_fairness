import os
import csv
import math
import matplotlib.pyplot as plt


# This script plots the correlation of WGA vs FairEntropy for different K values across multiple seeds, with error bars.


base = "__results"
seeds = ["seed1", "seed2", "seed3"]

Ks = [1, 2, 4, 8, 16, 32, 64]
filename = "Worst_Accuracy_vs_IS_correlation.csv"

methods = ["Pearson", "Spearman", "Kendall"]

# raw[method][K] = [values over seeds]
raw = {m: {k: [] for k in Ks} for m in methods}

# --- Read all seeds ---
for seed in seeds:
    path = f"{base}/{seed}"
    dirs = {
        1:  f"{path}/K1/correlations",
        2:  f"{path}/K2/correlations",
        4:  f"{path}/K4/correlations",
        8:  f"{path}/K8/correlations",
        16: f"{path}/K16/correlations",
        32: f"{path}/K32/correlations",
        64: f"{path}/K64/correlations",
    }

    for k, folder in dirs.items():
        csv_path = f"{folder}/{filename}"
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = row["method"]
                if method in raw:
                    raw[method][k].append(float(row["corr"]))

# --- Helpers (mean/std) ---
def mean_std(vals):
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)  # sample std
    return mu, math.sqrt(var)

# --- Plot with uncertainties ---
plt.figure(figsize=(8, 6))

for method in methods:
    ks = []
    mus = []
    sigmas = []
    for k in Ks:
        mu, sigma = mean_std(raw[method][k])
        if mu is None:
            continue
        ks.append(k)
        mus.append(mu)
        sigmas.append(sigma)

    plt.errorbar(ks, mus, yerr=sigmas, marker="o", capsize=4, label=method)

plt.xlabel("K (number of clusters)")
plt.ylabel("Correlation (mean ± std over seeds)")
plt.title("Worst Accuracy vs FairEntropy — correlation vs K (3 seeds)")
plt.grid(True)
plt.legend()

out_path = f"{base}/Worst_Accuracy_vs_IS_correlation_uncertainty.png"
plt.savefig(out_path)
plt.show()
print("Saved:", out_path)
