import itertools
import json

import hdbscan
import pandas as pd
import umap
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def perform_hyp_search(df: pd.DataFrame) -> dict[str, float | int | str]:
    """
    Perform hyperparameter search for UMAP.

    :param df: input data
    :return: the best hyperparameters
    """
    X = df.to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    best_score = -1
    best_params = {}

    # Define hyperparameters to search
    n_neighbors_options = [2, 3, 5, 10, 30]
    min_dist_options = [0.0, 0.1, 0.25, 0.5, 0.8]
    metrics = ["euclidean", "manhattan", "cosine", "correlation", "hamming"]
    perms = itertools.product(n_neighbors_options, min_dist_options, metrics)

    for perm in tqdm(
        perms, total=len(n_neighbors_options) * len(min_dist_options) * len(metrics)
    ):
        n_neighbors, min_dist, metric = perm
        # Apply UMAP
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=100,
            n_jobs=-1,
        )
        reduced_data = umap_model.fit_transform(X)

        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        cluster_labels = clusterer.fit_predict(reduced_data)

        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(reduced_data, cluster_labels)
            if sil_score > best_score:
                best_score = sil_score
                best_params = {
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "metric": metric,
                }

    print(f"Best Silhouette Score: {best_score}")
    print(f"Best Parameters: {best_params}")
    return best_params


if __name__ == "__main__":
    prefix = "../../datasets.nosync/"
    for dataset in ["clinical", "nmr"]:
        df = pd.read_csv(prefix + f"{dataset}_train.csv")
        df.drop(["CENSORED", "upper_bound", "lower_bound"], axis=1, inplace=True)
        if dataset == "clinical":
            df = df.fillna(df.median())
        best_params = perform_hyp_search(df)
        with open(f"{dataset}_umap_hp.json", "w") as f:
            json.dump(best_params, f)
