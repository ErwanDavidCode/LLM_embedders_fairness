import os
import numpy as np
import pandas as pd
import scipy.optimize as optim
from helpers import *
from sklearn import preprocessing

# 1. Charger le CSV déjà get_dummies et ordonné, LFR veut du one hot
df = pd.read_csv("adult_balanced_one_hot.csv")

# Les features sont toutes les colonnes sauf les deux dernières
feature_cols = df.columns[:-2] #on encode aussi l'attribut sensible car il est présent dans en -3em position aussi
sensitive_col = df.columns[-2]
target_col = df.columns[-1]

# Pas besoin de get_dummies ici, c'est déjà fait
X_features = df[feature_cols]

# Construction de la matrice finale pour LFR
data = X_features.values.astype(float)
sensitive = df[sensitive_col].values.astype(float)  # déjà 0/1
y = df[target_col].values.astype(float)             # déjà 0/1

# Standardisation
data = preprocessing.scale(data)

# Séparation sensible / non-sensible
sensitive_idx = np.where(sensitive == 1)[0]
nonsensitive_idx = np.where(sensitive != 1)[0]
data_sensitive = data[sensitive_idx, :]
data_nonsensitive = data[nonsensitive_idx, :]
y_sensitive = y[sensitive_idx]
y_nonsensitive = y[nonsensitive_idx]

# Split train/test (ici tout en train pour l'embedding, adapte si besoin)
training_sensitive = data_sensitive
ytrain_sensitive = y_sensitive
training_nonsensitive = data_nonsensitive
ytrain_nonsensitive = y_nonsensitive

training = np.concatenate((training_sensitive, training_nonsensitive))
ytrain = np.concatenate((ytrain_sensitive, ytrain_nonsensitive))

# 2. Apprentissage LFR
k = 10
src= ''
if os.path.isfile(src): #if file exists to load existing parameters
    with open(src, 'rb') as f:
        rez = f.read().split('\n')[:-1]
    rez = np.array([float(r) for r in rez])
    print(LFR(rez, training_sensitive, training_nonsensitive, ytrain_sensitive, 
              ytrain_nonsensitive, k, 1e-4, 0.1, 1000, 0))
else:
    print('not loading')
    rez = np.random.uniform(size=data.shape[1] * 2 + k + data.shape[1] * k)

bnd = []
for i, k2 in enumerate(rez):
    if i < data.shape[1] * 2 or i >= data.shape[1] * 2 + k:
        bnd.append((None, None))
    else:
        bnd.append((0, 1))

rez = optim.fmin_l_bfgs_b(LFR, x0=rez, epsilon=1e-5, 
                          args=(training_sensitive, training_nonsensitive, 
                                ytrain_sensitive, ytrain_nonsensitive, k, 1e-4,
                                0.1, 1000, 0),
                          bounds = bnd, approx_grad=True, maxfun=20000, #150k classic
                          maxiter=20000)[0] #150k classic

# 3. Générer les embeddings pour toutes les lignes du CSV
embeddings = get_z(data, rez, k)  # shape: (n_samples, k)

# 4. Sauvegarder le CSV embbedé
# On garde l'attribut sensible et la cible à la fin pour correspondre à l'original
df_embedded = pd.DataFrame(embeddings, columns=[f"z_{i}" for i in range(k)], index=df.index)
df_embedded[sensitive_col] = df[sensitive_col].values
df_embedded[target_col] = df[target_col].values

df_embedded.to_csv("LFR_adult_balanced_embedded.csv", index=False)
print("CSV généré : LFR_adult_balanced_embedded.csv")


