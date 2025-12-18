#!/usr/bin/env python3
import argparse
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to the embeddings CSV file")
    parser.add_argument("--output", required=True, help="Directory to save clustering results")
    parser.add_argument("--name", required=True, help="Name of the current embedder")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters (K)")
    args = parser.parse_args()

    # Load embeddings
    df = pd.read_csv(args.csv_path)


    # Extract only embeddings (drop target and sensitive attributes added)
    idx = df.columns.get_loc("target")
    vectors = df.iloc[:, :idx].to_numpy(copy=False)


    print(f"[INFO] Clustering {args.csv_path} with K={args.k}, seed={args.seed} ...")

    # Run clustering
    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        batch_size=10000,
        n_init="auto"
    )
    cluster_ids = kmeans.fit_predict(vectors)

    # Create aligned result DataFrame
    result = pd.DataFrame({
        "cluster": cluster_ids.astype("int32")
    })

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Save cluster assignment file
    assign_path = os.path.join(args.output, f"{args.name}_clusters.csv")
    result.to_csv(assign_path, index=False)
    print(f"[INFO] Saved cluster assignments → {assign_path}")

    # Save cluster sizes with IDs
    cluster_counts = pd.Series(cluster_ids).value_counts().sort_index()
    cluster_sizes_df = pd.DataFrame({
        "cluster_id": cluster_counts.index,
        "number_of_samples": cluster_counts.values
    })
    size_output = os.path.join(args.output, f"{args.name}_cluster_size.csv")
    cluster_sizes_df.to_csv(size_output, index=False)
    print(f"[INFO] Saved cluster sizes → {size_output}")



    # Print summary
    print("\n[INFO] Cluster sizes:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} samples")

    # Silhouette scores
    if args.k > 1 and len(vectors) > args.k:  # silhouette needs at least 2 clusters
        sil_samples = silhouette_samples(vectors, cluster_ids)
        sil_global = silhouette_score(vectors, cluster_ids)
        print(f"\n[INFO] Global silhouette score: {sil_global:.4f}")

        sil_per_cluster = pd.DataFrame({
            "cluster": cluster_ids,
            "silhouette": sil_samples
        }).groupby("cluster")["silhouette"].mean()

        print("[INFO] Silhouette score per cluster:")
        for cluster, score in sil_per_cluster.items():
            print(f"  Cluster {cluster}: {score:.4f}")
    else:
        print("[INFO] Silhouette score skipped (K=1 or not enough samples)")

    # Create completion flag
    flag_dir = "./completion_flags/step2_clustering"
    os.makedirs(flag_dir, exist_ok=True)
    flag_file = os.path.join(flag_dir, f"{args.name}_clusters.done")
    with open(flag_file, "w") as f:
        f.write("done\n")

if __name__ == "__main__":
    main()
