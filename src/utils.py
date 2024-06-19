import ast
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error


def convert_to_dmatrix(
    X: pd.DataFrame | np.ndarray,
    y_lower_bound: pd.Series | np.ndarray,
    y_upper_bound: pd.Series | np.ndarray,
) -> xgb.DMatrix:
    """
    Convert the data to a DMatrix.

    :param X: dataframe containing the features.
    :param y_lower_bound: lower bound of the event time.
    :param y_upper_bound: upper bound of the event time.
    :return: DMatrix of the data.
    """
    dtrain = xgb.DMatrix(X)
    dtrain.set_float_info("label_lower_bound", y_lower_bound)
    dtrain.set_float_info("label_upper_bound", y_upper_bound)
    return dtrain


def c_index(
    y_test_lower: np.ndarray, y_test_upper: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """
    Calculate C-index, censoring accuracy, and mean absolute error of the observed data.

    :param y_test: true labels.
    :param y_pred: predicted labels.
    :return: C-index of predicted data.
    """
    event_times = y_test_lower
    event_observed = (y_test_lower == y_test_upper).astype(int)
    try:
        mae_observed = mean_absolute_error(
            event_times[event_observed == 1], y_pred[event_observed == 1]
        )
    except ValueError:
        mae_observed = 100000
    censoring_accuracy = (
        event_times[event_observed == 0] <= y_pred[event_observed == 0]
    ).sum() / len(event_times[event_observed == 0])
    return (
        concordance_index(event_times, y_pred, event_observed),
        censoring_accuracy,
        float(mae_observed),
    )


def calculate_mcc(cindex: float, censoring_acc: float, mae_observed: float) -> float:
    """
    Calculate MCC score.

    The MCC score is a weighted average of the C-index, censoring accuracy and MAE observed.

    :param cindex: C-index of predicted data.
    :param censoring_acc: censoring accuracy of predicted data.
    :param mae_observed: mean absolute error of event data.
    :return: MCC score.
    """
    return max((cindex * 4 + censoring_acc + np.exp(-mae_observed) * 2) / 7, 0)


def load_feature_selection(file_name: str, X: pd.DataFrame) -> list:
    """
    Load feature selection results from a file.

    :param file_name: file name of the feature selection results.
    :param X: dataframe containing the features.
    :return: list of subsets of features.
    """
    with open(file_name, "r") as f:
        feature_selectors = json.load(f)
    features_selected = feature_selectors["featboost_aft"]["features_selected"]
    return [
        list(s)
        for s in {tuple(X.columns[i] for i in subset) for subset in features_selected}
    ]


def load_best_features(filename: str) -> list:
    """
    Load the best features from a file.

    :param filename: file name of the best features.
    :return: list of best features.
    """
    with open(filename, "r") as f:
        best_features = f.read()
    return ast.literal_eval(best_features)


def get_best_features_and_params(
    feature_selection_path: str, params_path: str
) -> tuple[dict, list]:
    """
    Load the best features and parameters from their respective files.

    :param feature_selection_path: path to the file containing the selected features.
    :param params_path: path to the parameters file.
    :return: tuple of best parameters and best features.
    """
    subset = load_best_features(feature_selection_path)
    with open(params_path, "r") as f:
        results = json.load(f)
    best_params = json.loads(max(results, key=results.get))
    xgb_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "n_jobs": -1,
        "tree_method": "hist",
        "booster": "gbtree",
        "grow_policy": "lossguide",
        "lambda": 0.01,
        "alpha": 0.02,
    }
    best_params.update(xgb_params)
    print("Best hyperparameters:", best_params)
    return best_params, subset


def bootstrap_split(
    censored_dataset: pd.Series,
    event_dataset: pd.Series,
    subset_fraction: tuple[float, float],
    replace: bool,
) -> np.ndarray:
    """
    Create a bootstrap sample of the dataset.

    :param censored_dataset: censored dataset.
    :param event_dataset: event dataset.
    :param subset_fraction: fraction of the dataset to sample (event, censored).
    :param replace: whether to sample with replacement.
    :return: indices of the sampled dataset.
    """
    subset_fraction_event, subset_fraction_censored = subset_fraction
    sampled_censored = censored_dataset.sample(
        int(len(censored_dataset) * subset_fraction_censored),
        replace=replace,
    )
    sampled_uncensored = event_dataset.sample(
        int(len(event_dataset) * subset_fraction_event),
        replace=replace,
    )
    return np.concatenate([sampled_censored.index, sampled_uncensored.index])
