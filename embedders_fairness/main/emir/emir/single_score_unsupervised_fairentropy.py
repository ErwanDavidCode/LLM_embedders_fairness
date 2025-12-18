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

        estimator = self.init_estimator(X, Y)
        mi, m, cond_cent = estimator.eval(X.float(), Y.float())
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





def load_group_embeddings(embedder_reference, cluster, seed, k):
    """
    Load embeddings for a specific cluster using cluster assignments stored in
    __results/seed{seed}/K{k}/clustering/{name}_clusters.csv.

    Robust to the case where embedding CSVs do NOT contain an 'id' column:
    - If clusters CSV has an 'id' column, those ids are treated as row indices
      into the embedding CSVs.
    - Otherwise, we assume cluster CSV rows are aligned with embedding CSV rows
      and use row positions.
    """
    name_ref, path_ref = embedder_reference

    cluster_file = f"__results/seed{seed}/K{k}/clustering/{name_ref}_clusters.csv"
    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    # Read cluster assignments
    df_clusters = pd.read_csv(cluster_file)

    if "cluster" not in df_clusters.columns:
        raise ValueError(f"'cluster' column not found in {cluster_file}")

    # Determine selected row indices for the requested cluster
    # Assume same ordering as embedding files
    mask = df_clusters["cluster"].to_numpy() == cluster
    selected_ids = mask.nonzero()[0]

    # Load reference embeddings
    df_ref = pd.read_csv(path_ref)
    ref_matrix = df_ref.iloc[:, :df_ref.columns.get_loc("target")].to_numpy(dtype="float32", copy=False)

    # Safety checks
    n_ref, _ = ref_matrix.shape
    if selected_ids.size == 0:
        # empty cluster -> return empty tensors
        return (torch.from_numpy(ref_matrix[selected_ids]), name_ref)
    if selected_ids.max() >= n_ref:
        raise IndexError(f"Selected index out of range: max(selected_ids)={selected_ids.max()} but ref rows={n_ref}")

    # Select rows by positional indexing
    ref_sel = ref_matrix[selected_ids]

    # Convert to torch tensors
    ref_tensor = torch.from_numpy(ref_sel)

    return (ref_tensor, name_ref)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, default="KNIFE")
    parser.add_argument("--estimator_config", type=str, default=os.path.join(os.path.dirname(__file__), "knife.yaml"))
    parser.add_argument("--name1", type=str, required=True)
    parser.add_argument("--input1", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=str, required=True,
                        help="Column grouping like 'SEX_RAC1P' or 'SEX_RAC1P_AGEP'")
    parser.add_argument("--output", type=str, required=True, help="Output directory for IS scores")
    args = parser.parse_args()


    # Initialize evaluator (assumes EmbedderEvaluator takes the args or adapt accordingly)
    evaluator = EmbedderEvaluator(args)

    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Loop clusters 0..k-1 (KMeans convention)
    for cluster in range(int(args.k)):
        print(f"\n=== Computing IS for cluster: {cluster} ===")

        # Load tensors for this cluster
        embs = load_group_embeddings(
            (args.name1, args.input1),
            cluster=cluster,
            seed=args.seed,
            k=int(args.k)
        )

        # #New
        emb_ref = embs
        metrics = evaluator(emb_ref, emb_ref) #embs = (torch.from_numpy(ref_sel), name_ref)
        metrics["IS_norm"] = metrics["H(Y)"].astype(float) / metrics["Y_dim"].astype(float) #because the IS value is scaling linearily with the size of the Y embedders. So we need to normalize it
        IS_score = metrics.pivot_table(index="X", columns="Y", values="IS_norm")

    
        #Sanity check
        print(f"Size of the embeddings for {metrics['Y'].iat[0]} : {int(metrics['Y_dim'].iat[0])}")
        print(metrics.loc[0, ["X","Y","I(X->Y)","H(Y)","H(Y|X)","Y_dim"]])
        print(metrics["I(X->Y)"].iat[0], metrics["H(Y)"].iat[0]-metrics["H(Y|X)"].iat[0])


        # Save IS score for this cluster
        base_filename = f"cluster{cluster}__{args.name1}"
        is_path = os.path.join(args.output, f"{base_filename}.csv")
        IS_score.to_csv(is_path)
        print(f"✅ IS saved: {is_path}")
        

    # Create completion flag in the requested hierarchical location
    flag_dir = "./completion_flags/step3_is_score"
    os.makedirs(flag_dir, exist_ok=True)
    flag_file = os.path.join(flag_dir, f"{args.name1}.done")
    with open(flag_file, "w") as f:
        f.write("done\n")
    print(f"Flag written: {flag_file}")
    print("\n")
    print("\n")


