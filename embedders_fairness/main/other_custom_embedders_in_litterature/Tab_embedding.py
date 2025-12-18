# TabICN embedding
import pandas as pd
import numpy as np
from tabicn import TabICNClassifier

# Charger les données
df = pd.read_csv("adult_balanced_one_hot.csv")
feature_cols = df.columns[:-2]
sensitive_col = df.columns[-2]
target_col = df.columns[-1]

X = df[feature_cols].values.astype(np.float32)
sensitive_feature = df[sensitive_col].values
y = df[target_col].values

# Instancier TabICN avec une grande dimension d'embedding (ex: 128)
clf_icn = TabICNClassifier(latent_dim=128)
clf_icn.fit(X, y)
X_tabicn = clf_icn.transform(X)  # Embedding TabICN

df_tabicn = pd.DataFrame(X_tabicn, columns=[f"tabicn_{i}" for i in range(X_tabicn.shape[1])], index=df.index)
df_tabicn[sensitive_col] = sensitive_feature
df_tabicn[target_col] = y
df_tabicn.to_csv("TabICN_adult_balanced_embedded.csv", index=False)
print("CSV généré : TabICN_adult_balanced_embedded.csv")