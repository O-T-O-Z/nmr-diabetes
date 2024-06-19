import json
import logging

import numpy as np
import pandas as pd
from deepsurv import DeepSurv
from numpy import float32, int32

from src.runner_functions import SurvivalDataset
from utils import load_best_features

logging.basicConfig(level=logging.INFO)


def create_data(
    X: pd.DataFrame,
    y_lower_bound: pd.Series,
    y_upper_bound: pd.Series,
    best_features: list,
) -> dict:
    """
    Create a dictionary with the data.

    :param X: features.
    :param y_lower_bound: lower bound of the survival time.
    :param y_upper_bound: upper bound of the survival time.
    :param best_features: features to use.
    :return: dictionary with the data.
    """
    return {
        "x": X[best_features].values.astype(float32),
        "t": (y_lower_bound == y_upper_bound).values.astype(float32),
        "e": y_lower_bound.values.astype(int32),
    }


def load_datasets(ds_name: str) -> tuple[dict, dict, dict, int]:
    """
    Load the datasets as a dictionary.

    :param ds_name: name of the dataset.
    :return: train, validation and test data.
    """
    train_dataset = SurvivalDataset()
    train_dataset.load_data(f"{ds_name}_train.csv")
    test_dataset = SurvivalDataset()
    test_dataset.load_data(f"{ds_name}_test.csv")

    if ds_name in ["clinical", "nmr"]:
        best_features = load_best_features(f"../results_fs/{ds_name}_best_features.txt")
    else:
        best_features = load_best_features(
            "../results_fs/clinical_best_features.txt"
        ) + load_best_features("../results_fs/nmr_best_features.txt")
    train_dataset.split()

    train_data = create_data(
        train_dataset.X_train,
        train_dataset.y_lower_bound_train,
        train_dataset.y_upper_bound_train,
        best_features,
    )
    valid_data = create_data(
        train_dataset.X_val,
        train_dataset.y_lower_bound_val,
        train_dataset.y_upper_bound_val,
        best_features,
    )
    test_data = create_data(
        test_dataset.X,
        test_dataset.y_lower_bound,
        test_dataset.y_upper_bound,
        best_features,
    )

    return train_data, valid_data, test_data, len(best_features)


def load_hyperparams(file_name: str, n_features: int) -> dict:
    """
    Load the hyperparameters from a file.

    :param file: file to load the hyperparameters from.
    :param n_features: number of features.
    :return: hyperparameters.
    """
    with open(file_name, "r") as f:
        params = json.load(f)

    params["learning_rate"] = 10 ** params["learning_rate"]
    params["hidden_layers_sizes"] = [
        int(params["num_nodes"]) for _ in range(int(params["num_layers"]))
    ]
    del params["num_layers"]
    del params["num_nodes"]
    params["batch_norm"] = True
    params["standardize"] = False
    params["activation"] = "rectify"
    params["n_in"] = n_features
    return params


if __name__ == "__main__":
    for ds_name in ["full", "clinical", "nmr"]:
        train_data, valid_data, test_data, n_features = load_datasets(ds_name)
        hyperparams = load_hyperparams(
            f"../results_models/deepsurv/{ds_name}_best_params.json", n_features
        )
        res = {"DeepSurv": []}
        for random_state in [42, 55, 875]:
            np.random.seed(random_state)
            network = DeepSurv(**hyperparams)

            n_epochs = 1000
            log = network.train(
                train_data,
                valid_data,
                n_epochs=n_epochs,
                validation_frequency=50,
                verbose=True,
            )
            res["DeepSurv"].append(network.get_concordance_index(**test_data))
            results_df = pd.DataFrame(res)
            print(results_df)
        results_df = pd.DataFrame(res)
        results_df.to_csv(f"../results_models/deepsurv/{ds_name}_deepsurv_results.csv")
