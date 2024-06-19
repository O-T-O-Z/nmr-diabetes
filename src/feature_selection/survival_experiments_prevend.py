import json
import os

import numpy as np
import pandas as pd
from featboostx.test_utils import (
    run_eval_survival,
    train_boruta_survival,
    train_featboost,
    train_xgb_survival,
)
from featboostx.xgb_survival_regressor import XGBSurvivalRegressor
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from feature_selection.utils import reduce
from visualization import plot_feature_selectors

SAVE_PATH = "../../results_fs"


def run_experiment(name: str, X: np.array, y: pd.DataFrame, params: dict) -> None:
    """
    Run the feature selection experiment by running three times 10-fold cross-validation.

    :param name: name of the dataset
    :param X: input data
    :param y: labels
    :param params: the best hyperparameters for the XGBSurvivalRegressor
    """
    censored = np.array(y["CENSORED"].copy())
    y = y.drop(["CENSORED"], axis=1)
    y = np.array(y)[:, ::-1]

    print(f"Running experiment {name}")

    feature_selectors = {
        k: {"cindex_per_fold": [], "features_selected": [], "feature_imp": []}
        for k in ["featboost_aft", "xgb", "boruta"]
    }

    for k_fold_seed in tqdm([84, 110, 1750]):
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=k_fold_seed)
        for train_index, test_index in tqdm(folds.split(X, censored)):  # type: ignore
            for selector, func in [
                ("boruta", train_boruta_survival),
                ("xgb", train_xgb_survival),
                ("featboost_aft", train_featboost),
            ]:
                print(f"Running feature selection with {selector}")
                if func == train_featboost:
                    features, importances = func(
                        X[train_index],
                        y[train_index],
                        [
                            XGBSurvivalRegressor(**params),
                            XGBSurvivalRegressor(**params),
                        ],
                        "c_index",
                    )
                else:
                    features, importances = func(
                        X[train_index], y[train_index], XGBSurvivalRegressor(**params)
                    )

                if (
                    importances.shape[0] == X.shape[1]
                    and np.unique(importances).shape[0] == 1
                ):
                    # if the importances are the same, use all the features
                    features = np.where(X[0, :])[0]

                importances = [float(i) for i in list(importances)]
                features = [int(f) for f in list(features)]
                # limit the number of features to 100
                if len(features) > 100:
                    features = features[:100]
                    importances = importances[:100]

                feature_selectors[selector]["cindex_per_fold"].append(
                    run_eval_survival(
                        X[train_index],
                        X[test_index],
                        y[train_index],
                        y[test_index],
                        features,
                        XGBSurvivalRegressor(**params),
                    )
                )
                feature_selectors[selector]["features_selected"].append(features)
                feature_selectors[selector]["feature_imp"].append(importances)
                with open(f"{SAVE_PATH}/{name}_featureselection.json", "w") as f:
                    json.dump(feature_selectors, f)

    feature_selectors = reduce(feature_selectors)
    with open(f"{SAVE_PATH}/{name}_featureselection.json", "w") as f:
        json.dump(feature_selectors, f)
    plot_feature_selectors(feature_selectors, name, SAVE_PATH)


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
        run_experiment(dataset, X, y, best_params_aft)
