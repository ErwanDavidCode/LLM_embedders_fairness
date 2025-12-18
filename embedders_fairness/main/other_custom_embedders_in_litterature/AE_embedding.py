import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Chargement des données
df = pd.read_csv("adult_balanced_one_hot.csv")
feature_cols = df.columns[:-2]
sensitive_col = df.columns[-2]
target_col = df.columns[-1]

X = df[feature_cols].values.astype(np.float32)
sensitive_feature = df[sensitive_col].values
y = df[target_col].values

# Standardisation (important pour PCA et AutoEncoder)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# 2. PCA
n_components = min(64, X.shape[1])  # Choix arbitraire, adapte si besoin
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=[f"pca_{i}" for i in range(X_pca.shape[1])], index=df.index)
df_pca[sensitive_col] = sensitive_feature
df_pca[target_col] = y
df_pca.to_csv("PCA_adult_balanced_embedded.csv", index=False)
print("CSV généré : PCA_adult_balanced_embedded.csv")



# 3. AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

input_dim = X_scaled.shape[1]
latent_dim = min(32, input_dim)
ae = AutoEncoder(input_dim, latent_dim)
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Entraînement rapide (10 epochs, adapte si besoin)
ae.train()
epochs = 100
for epoch in range(100):
    for batch in loader:
        x_batch = batch[0]
        optimizer.zero_grad()
        x_rec = ae(x_batch)
        loss = loss_fn(x_rec, x_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Extraction des embeddings
ae.eval()
with torch.no_grad():
    X_ae = ae.encoder(X_tensor).numpy()

df_ae = pd.DataFrame(X_ae, columns=[f"ae_{i}" for i in range(X_ae.shape[1])], index=df.index)
df_ae[sensitive_col] = sensitive_feature
df_ae[target_col] = y
df_ae.to_csv("AutoEncoder_adult_balanced_embedded.csv", index=False)
print("CSV généré : AutoEncoder_adult_balanced_embedded.csv")