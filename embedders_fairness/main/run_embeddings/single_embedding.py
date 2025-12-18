###
# Génération du CSV d'embeddings BERT à partir de adult_balanced.csv déjà get_dummies
###

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import pandas as pd
from collections import defaultdict
import os


# Conversion en texte pour chaque ligne (pour BERT)
def row_to_text(row):
    return ", ".join([f"{col}: {val}" for col, val in row.items()])



### Create mapping
def create_dic(path):
    """
    FOR THE ACS DATASET, create one phrase per info => good
    create basic dic to map the code to its description"""
    
    # Create dico with as many sub dic as we want : fluid architecture
    mapping = defaultdict(dict)

    with open(path, encoding="utf-8") as f:
        for line in f:
            # Ignore les lignes vides ou mal formées
            line = line.strip()
            if not line or line.startswith('"NAME'):
                continue
            
            # La vraie ligne CSV est avant le premier point-virgule
            parts_raw = line.split(";")[0]
            split_parts = parts_raw.split(",", maxsplit=6) #isolating last element because contains some ","
            row = [p.replace('""', '"').strip('"') for p in split_parts] # enlever les guillemets doubles et simples
            
            #name of the ID (ex "OCCP")
            id_key = row[1]
            #num value for the code if possible (ex "0010" -> "10", "000" -> "0")
            value = row[4].lstrip("0")
            if value == "":
                value = "0"  # ex: "000" -> 0

            #textual description
            description = row[-1]
            
            mapping[id_key][value] = description
    

    # convert into classic dic
    mapping = dict(mapping)
    return mapping


### Create sentences
def generate_row_description_acs(row, mapping):
    """FOR THE ACS DATASET, one phrase per info => clear and descriptive for embeddings"""
    
    def get_desc(col):
        if col not in row.index:
            return ""
        try:
            val_int = int(row[col])
        except:
            val_int = row[col]
        
        col_map = mapping.get(str(col), {})

        # 1. Si val_int existe dans le mapping
        if str(val_int) in col_map.keys():
            return col_map[str(val_int)]
        # 2. Sinon, si "b" existe dans le mapping
        elif "b" in col_map.keys():
            return col_map["b"]
        # 3. Sinon fallback
        else:
            return f"unknown {col} '{row[col]}'"


    # Genre et pronoms
    sex = get_desc('SEX')
    pronoun = "He" if sex == "Male" else "She"
    possessive = "His" if sex == "Male" else "Her"

    age_str = f"{int(row['AGEP'])}"
    hours_str = f"{int(row['WKHP'])}" if 'WKHP' in row else "unknown"
    income_str = f"{int(row['PINCP'])}" if 'PINCP' in row else "unknown"

    phrases = {
        "AGEP": f"{pronoun} is {age_str} years old.",
        "SEX": f"{pronoun} identifies as {sex.lower()}.",
        "RAC1P": f"{pronoun} is {get_desc('RAC1P').lower()}.",
        "MAR": f"{pronoun} is {get_desc('MAR').lower()}.",
        "RELP": f"{possessive} relationship to the householder is {get_desc('RELP').lower()}.",
        "POBP": f"{pronoun} was born in {get_desc('POBP').lower()}.",
        "SCHL": f"{pronoun} has completed {get_desc('SCHL').lower()}.",
        "COW": f"{pronoun} works as {get_desc('COW').lower()}.",
        "OCCP": f"{possessive} occupation involves {get_desc('OCCP').lower()}.",
        "WKHP": f"{pronoun} usually works {hours_str} hours per week.",

        "ST": f"{pronoun} lives in {get_desc('ST').lower()}.",
        "NATIVITY": f"{possessive} nativity is {get_desc('NATIVITY').lower()}.",
        "CIT": f"{possessive} citizenship status is {get_desc('CIT').lower()}.",
        "MIG": f"Did {pronoun.lower()} live in the same house one year ago: {get_desc('MIG').lower()}.",
        "MIL": f"Concerning military service, {pronoun.lower()} is {get_desc('MIL').lower()}.",
        "ESR": f"{pronoun} is {get_desc('ESR').lower()}.",
        "PINCP": f"{pronoun} earns {income_str} dollars annually.",
        "ESP": f"{possessive} parents are in the labor force as follows: {get_desc('ESP').lower()}.",
        "ANC": f"{pronoun} has ancestry recode {get_desc('ANC').lower()}.",
        "DIS": f"{pronoun} has a disability." if get_desc('DIS').lower() == "with a disability" else f"{pronoun} has no disability.",
        "DEAR": f"{pronoun} has hearing difficulty." if get_desc('DEAR').lower() == "yes" else f"{pronoun} has normal hearing.",
        "DEYE": f"{pronoun} has vision difficulty." if get_desc('DEYE').lower() == "yes" else f"{pronoun} has normal vision.",
        "DREM": f"{pronoun} has cognitive difficulty." if get_desc('DREM').lower() == "yes" else f"{pronoun} has no cognitive difficulty.",
        "FER": f"{pronoun} gave birth in the past 12 months." if get_desc('FER').lower() == "yes" else f"{pronoun} did not give birth in the past 12 months.",
    }

    final_text = " ".join([phrases[f] for f in phrases.keys() if f in row])
    return final_text




# --------------------------------- MAINimport os

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--name_model", type=str, required=True, help="Name of the model")
parser.add_argument("--model", type=str, required=True, help="Path to model snapshot")
parser.add_argument("--data_path", type=str, required=True, help="Path to the future embeddings")
parser.add_argument("--raw_data_path", type=str, required=True, help="Path to the the raw data")
args = parser.parse_args()

# Load dictionary and dataset (with target column)
mapping = create_dic("../../Data/datasets/PUMS_Data_Dictionary_2018.csv") #ATTENTION : it is fixed
raw_file = args.raw_data_path
df = pd.read_csv(raw_file)




# ----- WARNING: START OF THE MODIFYING SECTION BASED ON DATASET -----
# Separate columns to add them back as they are at the end of the features
target = df["target"]
SEX   = np.where(df["SEX"] == 1.0, "man", "woman")
RAC1P = np.where(df["RAC1P"] == 1.0, "white", "non_white")
AGEP  = np.where(df["AGEP"] < 43.0, "young", "old")
# Exclude columns that are not going to be encoded
exclude_col = ["target"]
features = df.drop(columns=exclude_col)
# ----- WARNING: END OF THE MODIFYING SECTION -----





# Convert each row of features into text description
X_text = features.apply(lambda row: generate_row_description_acs(row, mapping), axis=1)

import torch
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

# Encode all rows (list conversion is cheap compared to big DataFrame ops)
X_embed = model.encode(X_text.tolist(), show_progress_bar=True)

# Build final DataFrame = embeddings + target
df_embed = pd.DataFrame(
    X_embed,
    columns=[f"{args.name_model.lower()}_{i}" for i in range(X_embed.shape[1])],
    index=df.index
)




# ----- WARNING: START OF THE MODIFYING SECTION BASED ON DATASET -----
df_embed["target"] = target #TARGET NEEDS TO BE THE FIRST COLUMN OF ADDED COLUMNS
df_embed["SEX"] = SEX
df_embed["RAC1P"] = RAC1P
df_embed["AGEP"] = AGEP
# ----- WARNING: END OF THE MODIFYING SECTION -----





# Save as CSV
output_filename = f"{args.data_path}/{args.name_model}.csv"
df_embed.to_csv(output_filename, index=False)

print(f"CSV generated: {output_filename}")


# Création du flag à la fin du script
flag_dir = "./completion_flags/step1_embeddings"
os.makedirs(flag_dir, exist_ok=True)
flag_path = os.path.join(flag_dir, f"{args.name_model}.done")
with open(flag_path, "w") as f:
    f.write("done\n")