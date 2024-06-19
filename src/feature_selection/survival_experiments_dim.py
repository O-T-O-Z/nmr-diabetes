import json
import os
from typing import Union

import numpy as np
import pandas as pd
from featboostx.test_utils import run_eval_survival
from featboostx.xgb_survival_regressor import XGBSurvivalRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

from src.feature_selection.feat_utils import reduce
from visualization import plot_feature_selectors

SAVE_PATH = "../../results_fs"


def run_dim_reduction(
    X_scaled: np.ndarray,
    X_scaled_test: np.ndarray,
    reductor: Union[UMAP, PCA],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run dimensionality reduction on the data.

    :param X_scaled: scaled training data
    :param X_scaled_test: scaled test data
    :param reductor: the dimensionality reduction algorithm
    :param name: the name of the dataset
    :return: the reduced training and test data
    """
    if reductor == UMAP:
        with open(os.path.join(SAVE_PATH, f"{name}_umap_hp.json"), "r") as f:
            hp = json.load(f)
        reduced = reductor(
            n_components=100,
            random_state=0,
            transform_seed=0,
            **hp,
        )
    else:
        reduced = reductor(n_components=100, random_state=0)
    return reduced.fit_transform(X_scaled), reduced.transform(X_scaled_test)


def run_experiment_dim_reduction(
    name: str, X: np.array, y: pd.DataFrame, params: dict
) -> None:
    """
    Run the dimensionality reduction experiment by running three times 10-fold cross-validation.

    :param name: name of the dataset
    :param X: input data
    :param y: labels
    :param params: the best hyperparameters for the XGBSurvivalRegressor
    """
    censored = np.array(y["CENSORED"].copy())
    y = y.drop(["CENSORED"], axis=1)
    y = np.array(y)[:, ::-1]

    dim_reductors = {
        k: {"cindex_per_fold": [], "features_selected": [], "feature_imp": []}
        for k in ["pca", "umap"]
    }
    for k_fold_seed in tqdm([84, 110, 1750]):
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=k_fold_seed)
        for train_index, test_index in tqdm(folds.split(X, censored)):  # type: ignore
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X[train_index])
            X_scaled_test = scaler.transform(X[test_index])
            for dim_reductor, reductor in [
                ("pca", PCA),
                ("umap", UMAP),
            ]:
                print(f"Running dimensionality reduction with {dim_reductor}")
                X_reduced, X_reduced_test = run_dim_reduction(
                    X_scaled, X_scaled_test, reductor, name
                )
                features = list(range(100))

                dim_reductors[dim_reductor]["cindex_per_fold"].append(
                    run_eval_survival(
                        X_reduced,
                        X_reduced_test,
                        y[train_index],
                        y[test_index],
                        features,
                        XGBSurvivalRegressor(**params),
                    )
                )
                dim_reductors[dim_reductor]["features_selected"].append(features)
                dim_reductors[dim_reductor]["feature_imp"].append(
                    [1 / 100 for _ in range(100)]
                )

                with open(f"{SAVE_PATH}/{name}_dimreduction.json", "w") as f:
                    json.dump(dim_reductors, f)

    dim_reductors = reduce(dim_reductors)

    with open(f"{SAVE_PATH}/{name}_dimreduction.json", "w") as f:
        json.dump(dim_reductors, f)

    plot_feature_selectors(dim_reductors, name + "_dim", SAVE_PATH)


if __name__ == "__main__":
    with open(os.path.join(SAVE_PATH, "best_params_xgb_aft.json"), "r") as f:
        best_params_aft = json.load(f)
    prefix = "../../datasets.nosync/"

    for dataset in ["clinical", "nmr"]:
        df = pd.read_csv(prefix + f"{dataset}_train.csv")
        y = df[["CENSORED", "upper_bound", "lower_bound"]]
        X = df.drop(["CENSORED", "upper_bound", "lower_bound"], axis=1, inplace=False)
        if dataset == "clinical":
            X = X.fillna(X.median())
        X = X.to_numpy()
        run_experiment_dim_reduction(dataset, X, y, best_params_aft)
