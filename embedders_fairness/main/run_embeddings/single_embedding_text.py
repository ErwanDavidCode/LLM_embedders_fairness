# --------------------------------- MAIN
import os
import argparse
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--name_model", type=str, required=True, help="Name of the model")
parser.add_argument("--model", type=str, required=True, help="Path to model snapshot")
parser.add_argument("--data_path", type=str, required=True, help="Path to the future embeddings")
parser.add_argument("--raw_data_path", type=str, required=True, help="Path to the raw data")
args = parser.parse_args()


input_csv = args.raw_data_path


# Load dataset (CSV with "text" and "target")
df = pd.read_csv(input_csv)

# Separate text and target
X_text = df["text"]
target = df["target"]




print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Load SentenceTransformer model
print(f"\n=== Encoding with model: {args.name_model} ===")

def find_snapshot_path(model_root):
    snapshot_dir = os.path.join(model_root, "snapshots")
    if os.path.exists(snapshot_dir) and os.path.isdir(snapshot_dir):
        subdirs = os.listdir(snapshot_dir)
        if not subdirs:
            raise RuntimeError(f"No snapshot found in {snapshot_dir}")
        return os.path.join(snapshot_dir, subdirs[0])
    else:
        return model_root

model = SentenceTransformer(find_snapshot_path(args.model))

# Encode all rows
X_embed = model.encode(X_text.tolist(), show_progress_bar=True)

# Build final DataFrame = embeddings + target
df_embed = pd.DataFrame(
    X_embed,
    columns=[f"{args.name_model.lower()}_{i}" for i in range(X_embed.shape[1])],
    index=df.index
)
df_embed["target"] = target




# Save as CSV
os.makedirs(args.data_path, exist_ok=True)
output_filename = os.path.join(args.data_path, f"{args.name_model}.csv")
df_embed.to_csv(output_filename, index=False)

print(f"✅ CSV generated: {output_filename}")

# Create completion flag
flag_dir = "./completion_flags/step1_embeddings"
os.makedirs(flag_dir, exist_ok=True)
flag_path = os.path.join(flag_dir, f"{args.name_model}.done")
with open(flag_path, "w") as f:
    f.write("done\n")
