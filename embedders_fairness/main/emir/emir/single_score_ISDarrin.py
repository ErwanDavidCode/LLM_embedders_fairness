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



def load_group_embeddings(embedder_reference, embedder_to_compare):
    """
    Load embeddings for all rows from two embedding CSV files.
    This version does NOT perform any filtering or selection,
    it simply returns all embeddings for both references.
    """

    name_ref, path_ref = embedder_reference
    name_cmp, path_cmp = embedder_to_compare

    # Load embeddings CSVs
    df_ref = pd.read_csv(path_ref)
    df_cmp = pd.read_csv(path_cmp)

    # Check mandatory "target" column presence
    if "target" not in df_ref.columns:
        raise ValueError(f"Column 'target' missing in {path_ref}")
    if "target" not in df_cmp.columns:
        raise ValueError(f"Column 'target' missing in {path_cmp}")

    # Extract embeddings only (all columns before "target")
    ref_matrix = df_ref.iloc[:, :df_ref.columns.get_loc("target")].to_numpy(dtype="float32", copy=False)
    cmp_matrix = df_cmp.iloc[:, :df_cmp.columns.get_loc("target")].to_numpy(dtype="float32", copy=False)

    # Return ALL rows (no filtering applied)
    return (torch.from_numpy(ref_matrix), name_ref), (torch.from_numpy(cmp_matrix), name_cmp)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, default="KNIFE")
    parser.add_argument("--estimator_config", type=str, default=os.path.join(os.path.dirname(__file__), "knife.yaml"))
    parser.add_argument("--name1", type=str, required=True)
    parser.add_argument("--input1", type=str, required=True)
    parser.add_argument("--name2", type=str, required=True)
    parser.add_argument("--input2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Output directory for IS scores")
    args = parser.parse_args()

    evaluator = EmbedderEvaluator(args)
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("\n" * 3)

    # Calculating IS score for this single pair
    embs = load_group_embeddings(
        (args.name1, args.input1),
        (args.name2, args.input2),
    )

    # #New
    emb_ref, emb_cmp = embs
    metrics = evaluator(emb_ref, emb_cmp) #embs = [(torch.from_numpy(ref_sel), name_ref), (torch.from_numpy(cmp_sel), name_cmp)]
    metrics["IS_norm"] = metrics["I(X->Y)"].astype(float) / metrics["Y_dim"].astype(float) #because the IS value is scaling linearily with the size of the Y embedders. So we need to normalize it
    IS_score = metrics.pivot_table(index="X", columns="Y", values="IS_norm")


    print("\n")
    print(f"recorded_marg_ent: {evaluator.estimator.recorded_marg_ent}")
    print("\n")
    print(f"recorded_cond_ent: {evaluator.estimator.recorded_cond_ent}")
    print("\n")

    # Save for marg loss
    base_filename = f"{args.name1}_VS_{args.name2}"
    is_path = os.path.join(f"{args.output}/loss_marg", f"{base_filename}.csv")
    df_loss = pd.DataFrame(evaluator.estimator.recorded_marg_ent, columns=["epoch"])
    df_loss.to_csv(is_path)
    print(f"✅ Loss marg saved: {is_path}")

    # Save for cond loss
    base_filename = f"{args.name1}_VS_{args.name2}"
    is_path = os.path.join(f"{args.output}/loss_cond", f"{base_filename}.csv")
    df_loss = pd.DataFrame(evaluator.estimator.recorded_cond_ent, columns=["epoch"])
    df_loss.to_csv(is_path)
    print(f"✅ Loss cond saved: {is_path}")


    #Sanity check
    print(f"Size of the embeddings for {metrics['Y'].iat[0]} : {int(metrics['Y_dim'].iat[0])}")
    print(metrics.loc[0, ["X","Y","I(X->Y)","H(Y)","H(Y|X)","Y_dim"]])
    print(metrics["I(X->Y)"].iat[0], metrics["H(Y)"].iat[0]-metrics["H(Y|X)"].iat[0])
    print(metrics["X"].iat[0] == args.name1 and metrics["Y"].iat[0] == args.name2)


    # Save IS
    base_filename = f"{args.name1}_VS_{args.name2}"
    is_path = os.path.join(args.output, f"{base_filename}.csv")
    IS_score.to_csv(is_path)
    print(f"✅ IS saved: {is_path}")


    # Completion flag
    flag_dir = "./completion_flags/step3_is_score"
    os.makedirs(flag_dir, exist_ok=True)
    flag_file = os.path.join(flag_dir, f"{args.name1}__vs__{args.name2}.done")
    with open(flag_file, "w") as f:
        f.write("done\n")
    print(f"Flag written: {flag_file}")



