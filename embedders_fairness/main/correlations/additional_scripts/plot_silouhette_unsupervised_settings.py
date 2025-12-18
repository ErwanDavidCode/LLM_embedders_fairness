import os
import re
import math
import matplotlib.pyplot as plt


# This script plots the silhouette scores for different K values across multiple seeds.


base = "__results"
seeds = ["seed1", "seed2", "seed3"]

Ks = [1, 2, 4, 8, 16, 32, 64]

# --- Regex (comme ton script silhouette) ---
embedder_pattern = re.compile(r"^(.*)_\d+\.out$")
score_pattern = re.compile(r"\[INFO\] Global silhouette score:\s+([0-9.]+)")

# raw[embedder][K] = [scores over seeds]
raw = {}

# --- Read all seeds ---
for seed in seeds:
    path = f"{base}/{seed}"
    dirs = {k: f"{path}/K{k}/clustering/logs" for k in Ks}

    for k, folder in dirs.items():
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            if not (fname.endswith(".out") and not fname.startswith("ALL_")):
                continue

            m = embedder_pattern.match(fname)
            if not m:
                continue
            embedder = m.group(1)

            with open(os.path.join(folder, fname), "r") as f:
                content = f.read()

            sm = score_pattern.search(content)
            if not sm:
                continue

            score = float(sm.group(1))
            raw.setdefault(embedder, {}).setdefault(k, []).append(score)

# --- Helpers ---
def mean_std(vals):
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return mu, math.sqrt(var)

# --- Plot ---
plt.figure(figsize=(8, 6))

lines = []
labels = []

for embedder, per_k in raw.items():
    ks, mus, sigmas = [], [], []
    for k in Ks:
        mu, sigma = mean_std(per_k.get(k, []))
        if mu is None:
            continue
        ks.append(k)
        mus.append(mu)
        sigmas.append(sigma)

    if not ks:
        continue

    cont = plt.errorbar(ks, mus, yerr=sigmas, marker="o", capsize=4, label=embedder)
    lines.append(cont.lines[0])  # line handle
    labels.append(embedder)

# --- Trier la légende par score moyen à K=2 ---
scores_k2 = {e: mean_std(raw[e].get(2, []))[0] for e in raw}
sorted_pairs = sorted(scores_k2.items(), key=lambda x: (x[1] is not None, x[1]), reverse=True)

sorted_labels = [p[0] for p in sorted_pairs if p[0] in labels]
sorted_lines = [lines[labels.index(lbl)] for lbl in sorted_labels]

plt.xlabel("K (number of clusters)")
plt.ylabel("Global silhouette score (mean ± std over seeds)")
plt.title("Global silhouette score depending on K (3 seeds)")
plt.legend(sorted_lines, sorted_labels, fontsize="small")
plt.grid(True)

out_path = f"{base}/silhouette_uncertainty.png"
plt.savefig(out_path)
plt.show()
print("Saved:", out_path)
