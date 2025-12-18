"""Universal wrapper for the KNIFE estimator, adaptations of these wrappers can be found in the domain specific scripts."""

import os
import argparse
from itertools import product
from typing import List, Tuple
import yaml

import numpy as np

import pandas as pd
import torch
from tqdm import tqdm

import json
from emir.emir.estimators import KNIFEEstimator, KNIFEArgs


def get_config_cls_estimator(estimator_config, estimator_name):
    if estimator_name == "KNIFE":
        return KNIFEArgs(**estimator_config), KNIFEEstimator
    raise ValueError(f"Estimator {estimator_name} not found")


class EmbedderEvaluator:
    def __init__(self, args: argparse.Namespace):
        """
        Wrapper for the emebdding evaluation
        :param args: argparse.Namespace object, contains the configuration for
        the etimator class, whose name is referenced in args.estimator and the
        configuration file in args.estimator_config
        """
        self.args = args
        self.config, self.estimator_cls = get_config_cls_estimator(
            self.get_estimator_config(), args.estimator
        )
        self.estimator = None
        # Metrics
        self.metrics = pd.DataFrame()

    def get_estimator_config(self):
        """
        Load the estimator configuration from the yaml file
        :return:
        """
        if not os.path.exists(self.args.estimator_config):
            return {}
        with open(self.args.estimator_config, "r") as f:
            estimator_args = yaml.safe_load(f)
        return estimator_args

    def init_estimator(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Initialize the estimator class
        :param X: Embeddings of the models simulating the other one
        :param Y: Embeddings of the model to be simulated
        :return:
        """
        return self.estimator_cls(
            self.config,
            X.shape[1],
            Y.shape[1],
        )

    def estimate_model_pair(
        self, pair: Tuple[Tuple[torch.Tensor, str], Tuple[torch.Tensor, str]]
    ):
        """
        Estimate the mutual information between two models.
        :param pair: Tuple of two tuples, each containing the embeddings of the models and their names
        :return:
        """
        X, X_name = pair[0]
        Y, Y_name = pair[1]

        self.estimator = self.init_estimator(X, Y)
        mi, m, cond_cent = self.estimator.eval(X.float(), Y.float())
        row = {
            "X": X_name,
            "Y": Y_name,
            "X_dim": X.shape[1],
            "Y_dim": Y.shape[1],
            "I(X->Y)": mi,
            "H(Y)": m,
            "H(Y|X)": cond_cent,
        }
        if self.metrics.shape[0] == 0:
            self.metrics = pd.DataFrame(columns=row.keys())
        self.metrics.loc[self.metrics.shape[0]] = row

    def __call__(
        self,
        X_pair: List[Tuple[torch.Tensor, str]],
        Y_pair: List[Tuple[torch.Tensor, str]],
    ):
        """
        MODIFIED
        Run the evaluation of the embeddings
        :param X: A list of tuples containing the embeddings of the models simulating the other ones and their names.
        :param Y: A list of tuples containing the embeddings of the models to be simulated and their names.
        :return:
        """
        # #Old
        # results = list(
        #     tqdm(
        #         map(self.estimate_model_pair, [(X_pair[0], Y_pair[0])]),
        #         total=1,
        #     )
        # )

        #New
        self.metrics = pd.DataFrame()
        self.estimate_model_pair((X_pair, Y_pair))  # ← ref -> cmp


        return self.metrics





def load_group_embeddings_from_columns(embedder_reference,
                                       group_columns, filters):
    """
    Load embeddings for specific groups defined by categorical columns.
    Select rows based on filters (col -> value).
    """

    name_ref, path_ref = embedder_reference

    # Load embeddings CSVs
    df_ref = pd.read_csv(path_ref)

    # Check mandatory columns
    required_cols = ["target"] + group_columns
    for col in required_cols:
        if col not in df_ref.columns:
            raise ValueError(f"Column '{col}' missing in {path_ref}")

    # Apply filtering mask according to filters dict
    mask = pd.Series(True, index=df_ref.index)
    for col, value in filters.items():
        mask &= df_ref[col] == value

    selected_ids = mask[mask].index.to_numpy()
    if selected_ids.size == 0:
        print(f"⚠️ No rows found for filters={filters}")
        return (torch.empty((0, 0)), name_ref)

    # Extract embeddings only
    ref_matrix = df_ref.iloc[:, :df_ref.columns.get_loc("target")].to_numpy(dtype="float32", copy=False)


    # Select rows
    ref_sel = ref_matrix[selected_ids]

    return (torch.from_numpy(ref_sel), name_ref)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, default="KNIFE")
    parser.add_argument("--estimator_config", type=str, default=os.path.join(os.path.dirname(__file__), "knife.yaml"))
    parser.add_argument("--name1", type=str, required=True)
    parser.add_argument("--input1", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=str, required=True,
                        help="Column grouping like 'SEX_RAC1P' or 'SEX_RAC1P_AGEP'")
    parser.add_argument("--output", type=str, required=True, help="Output directory for scores")
    args = parser.parse_args()

    evaluator = EmbedderEvaluator(args)
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("\n")
    print("\n")
    print("\n")

    # Grouping columns
    group_columns = args.k.split("_") #ex: ["SEX", "RAC1P"]

    # Load one CSV (reference) just to enumerate unique values per group column
    df_ref = pd.read_csv(args.input1)

    # Build list of unique values per column
    unique_values = {col: pd.unique(df_ref[col]) for col in group_columns}

    # Cartesian product of all group values (dynamic nested loops)
    for combo in product(*[unique_values[col] for col in group_columns]):
        filters = dict(zip(group_columns, combo))
        print(f"\n=== Computing IS for group {filters} ===")

        embs = load_group_embeddings_from_columns(
            (args.name1, args.input1),
            group_columns=group_columns,
            filters=filters
        )

        # #New
        emb_ref = embs
        metrics = evaluator(emb_ref, emb_ref) #embs = (torch.from_numpy(ref_sel), name_ref)
        metrics["IS_norm"] = metrics["H(Y)"].astype(float) / metrics["Y_dim"].astype(float) #I(X->Y) #H(Y) #because the IS value is scaling linearily with the size of the Y embedders. So we need to normalize it
        score = metrics.pivot_table(index="X", columns="Y", values="IS_norm")

    
        print("\n")
        print(f"Loss of the MLP predicting the parameters of the conditional probability: {evaluator.estimator.recorded_loss}")
        print("\n")
        print(f"recorded_marg_ent: {evaluator.estimator.recorded_marg_ent}")
        print("\n")
        print(f"recorded_cond_ent: {evaluator.estimator.recorded_cond_ent}")
        print("\n")


        #Sanity check
        print(f"Size of the embeddings for {metrics['Y'].iat[0]} : {int(metrics['Y_dim'].iat[0])}")
        print(metrics.loc[0, ["X","Y","I(X->Y)","H(Y)","H(Y|X)","Y_dim"]])
        print(metrics["I(X->Y)"].iat[0], metrics["H(Y)"].iat[0]-metrics["H(Y|X)"].iat[0])
        print(metrics["X"].iat[0] == args.name1)

        
        # Build filename including group values
        suffix = "".join(f"{col}{val}" for col, val in filters.items())
        base_filename = f"{suffix}__{args.name1}"
        is_path = os.path.join(args.output, f"{base_filename}.csv")
        score.to_csv(is_path)
        print(f"✅ IS saved: {is_path}")


    print("Saving flags")
    # Completion flags (one per embedder)
    flag_dir = "./completion_flags/step3_score"
    os.makedirs(flag_dir, exist_ok=True)
    flag_file = os.path.join(flag_dir, f"{args.name1}.done")
    with open(flag_file, "w") as f:
        f.write("done for all of the sensitive groups defined\n")
    print(f"Flag written: {flag_file}")
