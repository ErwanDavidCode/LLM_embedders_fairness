import os
import argparse
import pandas as pd
import numpy as np



def main(args):

    # Raw path
    raw_file = args.raw_data_path
    print("Dataset used : ", raw_file)
    print("Embedding used : ", args.input)

    # Load datasets
    df_raw = pd.read_csv(raw_file)
    df_embed = pd.read_csv(args.input)

    # Ensure same number of rows
    if len(df_raw) != len(df_embed):
        raise ValueError("Raw and embedded CSV do not have the same number of rows!")









    # ----- WARNING: START OF THE MODIFYING SECTION IF YOU WANT TO ADD FEATURES OF EXISTING EMBEDDINGS -----

    # Add columns (vectorized). Those columns are already added to the embedding via run_all_embeddings, this code is a "manual" example completion. Be aware of the values in the datasets.
    #TARGET STILL NEEDS TO BE THE FIRST COLUMN OF ADDED COLUMNS
    if "TARGET" not in df_embed.columns:
        df_embed["TARGET"] = np.where(df_raw["TARGET"].values >= 50000,True, False)

    if "SEX" not in df_embed.columns:
        df_embed["SEX"] = np.where(df_raw["SEX"].values == 1.0, "man", "woman")

    if "RAC1P" not in df_embed.columns:
        df_embed["RAC1P"] = np.where(df_raw["RAC1P"].values == 1.0, "white", "non_white")

    if "AGEP" not in df_embed.columns:
        df_embed["AGEP"] = np.where(df_raw["AGEP"].values < 43.0, "young", "old")
    
    # ----- WARNING: END OF THE MODIFYING SECTION -----





    # Overwrite embedded dataset with enriched version
    df_embed.to_csv(args.input, index=False)

    # Create completion flag
    flag_dir = "./completion_flags/step1_add_data"
    os.makedirs(flag_dir, exist_ok=True)
    flag_path = os.path.join(flag_dir, f"{args.model_name}.done")
    with open(flag_path, "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Embedded CSV file to enrich")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for the flag")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw data")
    args = parser.parse_args()
    main(args)

    print("done")
