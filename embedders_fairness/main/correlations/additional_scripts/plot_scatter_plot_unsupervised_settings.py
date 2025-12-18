import os
import pandas as pd
import matplotlib.pyplot as plt


# This script plots scatter plots of WGA vs FairEntropy for different K values in unsupervised settings.



# Répertoire de base
path = "__results/seed1"

# Fichiers d'entrée pour chaque K
files = {
    1: f"{path}/K1/correlations/Worst_Accuracy_vs_IS_summary.csv",
    2: f"{path}/K2/correlations/Worst_Accuracy_vs_IS_summary.csv",
    4: f"{path}/K4/correlations/Worst_Accuracy_vs_IS_summary.csv",
    8: f"{path}/K8/correlations/Worst_Accuracy_vs_IS_summary.csv",
    16: f"{path}/K16/correlations/Worst_Accuracy_vs_IS_summary.csv",
    32: f"{path}/K32/correlations/Worst_Accuracy_vs_IS_summary.csv"}

# Couleurs associées à chaque K
colors = {
    1: "tab:gray",
    2: "tab:blue",
    4: "tab:orange",
    8: "tab:green",
    16: "tab:red",
    32: "tab:purple",
    64: "tab:brown",
}

# Colonnes à utiliser (d'après ton CSV)
x_col = "IS_score_worst"
y_col = "Accuracy_worst"

plt.figure(figsize=(10, 7))

any_points = False  # pour vérifier qu'on a bien tracé quelque chose

for k, csv_path in files.items():
    if not os.path.exists(csv_path):
        print(f"[WARN] Fichier introuvable pour K={k} : {csv_path}")
        continue

    # Lecture du CSV tel que donné :
    # embedder,Accuracy_worst,IS_score_worst
    df = pd.read_csv(csv_path)

    # Nettoyage éventuel des noms de colonnes
    df.columns = [c.strip() for c in df.columns]

    # Vérification des colonnes
    if x_col not in df.columns or y_col not in df.columns or "embedder" not in df.columns:
        print(
            f"[WARN] Colonnes attendues manquantes dans {csv_path}.\n"
            f"Colonnes trouvées : {df.columns}"
        )
        continue

    # Optionnel : enlever un embedder spécifique
    # df = df[df["embedder"] != "SFR-Embedding-Mistral"]

    x = df[x_col]
    y = df[y_col]
    embedder_names = df["embedder"]


    color = colors.get(k, "gray")
    plt.scatter(x, y, c=color, label=f"K={k}", alpha=0.8)

    any_points = True

    # Annotation avec le nom de l'embedder
    for xi, yi, name in zip(x, y, embedder_names):
        plt.text(
            xi,
            yi,
            name,
            fontsize=8,
            ha="left",
            va="bottom"
        )

if not any_points:
    print("[INFO] Aucun point tracé – vérifier les chemins et les colonnes.")
else:
    plt.title("FairEntropy vs WGA for K = 1, 2, 4, 8, 16, 32")
    plt.xlabel("FairEntropy")
    plt.ylabel("WGA")
    plt.grid(True)
    plt.legend(title="Number of clusters K")
    plt.tight_layout()

    # Sauvegarde dans le dossier principal
    plt.savefig(f"{path}/Worst_IS_vs_Accuracy_allK.png", dpi=300)
    plt.show()
