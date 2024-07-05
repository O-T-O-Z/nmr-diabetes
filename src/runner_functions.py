import itertools
import json
import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from survival_dataset import SurvivalDataset
from utils import bootstrap_split, c_index, calculate_mcc, convert_to_dmatrix
from visualization import plot_shap, plot_survival_time, plot_training


def train_one_model(
    dataset: SurvivalDataset,
    xgbparams: dict,
    plot: bool = False,
    eval_metric: None | str = None,
    early_stopping_rounds: int = 50,
) -> xgb.Booster | float:
    """
    Train a single model on the given dataset.

    :param dataset: dataset to train the model on.
    :param xgbparams: parameters for the model.
    :param plot: whether to plot the training and validation losses, defaults to False.
    :param eval_metric: either 'cindex', 'mcc', or 'all', defaults to None.
    :param early_stopping_rounds: early stopping rounds, defaults to 50.
    :return: trained model or evaluation metric.
    """
    dtrain = convert_to_dmatrix(
        dataset.X_train, dataset.y_lower_bound_train, dataset.y_upper_bound_train
    )
    dvalid = convert_to_dmatrix(
        dataset.X_val, dataset.y_lower_bound_val, dataset.y_upper_bound_val
    )

    evals_result = {}
    clf = xgb.train(
        xgbparams,
        dtrain,
        num_boost_round=10000,
        early_stopping_rounds=early_stopping_rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
        evals_result=evals_result,
    )
    if plot:
        plot_training(evals_result)
    if eval_metric is not None:
        cindex, censoring_acc, mae_observed = c_index(
            dataset.y_lower_bound_val,
            dataset.y_upper_bound_val,
            clf.predict(dvalid),
        )
        if eval_metric == "cindex":
            return cindex
        if eval_metric == "mcc":
            return calculate_mcc(cindex, censoring_acc, mae_observed)
        if eval_metric == "all":
            return cindex, censoring_acc, mae_observed
    return clf


def test_model(
    train_dataset: SurvivalDataset,
    test_dataset: SurvivalDataset,
    best_features: list,
    best_params: dict,
    random_state: int,
    ds: str,
    early_stopping_rounds: int = 50,
    scale: float = 1.0,
    plot_path: str | None = None,
) -> tuple[float, float, float]:
    """
    Test the model on the given test dataset and plot SHAP.

    :param train_dataset: dataset to train the model on.
    :param test_dataset: dataset to test the model on.
    :param best_features: list of best features.
    :param best_params: best parameters for the model.
    :param random_state: random state for the test.
    :param ds: dataset to use.
    :param early_stopping_rounds: early stopping rounds, defaults to 50.
    :param scale: scales the predictions, defaults to 1.0.
    :param plot_path: path to save SHAP plot, defaults to None.
    :return: c-index, censoring accuracy, and MAE observed.
    """
    train_dataset.split(test_size=0.1, random_state=random_state)
    clf = train_one_model(
        train_dataset,
        best_params,
        early_stopping_rounds=early_stopping_rounds,
    )
    dtest = convert_to_dmatrix(
        test_dataset.X[best_features],
        test_dataset.y_lower_bound,
        test_dataset.y_upper_bound,
    )
    dtrain = convert_to_dmatrix(
        train_dataset.X[best_features],
        train_dataset.y_lower_bound,
        train_dataset.y_upper_bound,
    )
    predicted_times = clf.predict(dtest) * scale
    with open(os.path.join(plot_path, f"ys_train_{ds}.npy"), "wb") as f:
        np.save(f, clf.predict(dtrain) * scale)

    with open(os.path.join(plot_path, f"ys_test_{ds}.npy"), "wb") as f:
        np.save(f, predicted_times * scale)
    cindex, censoring_acc, mae_observed = c_index(
        test_dataset.y_lower_bound,
        test_dataset.y_upper_bound,
        predicted_times,
    )
    if ds == "full":
        plot_shap(clf, test_dataset, best_features, plot_path=plot_path)
        plot_survival_time(test_dataset, predicted_times, plot_path=plot_path)
    return cindex, censoring_acc, mae_observed


def test_model_bagging(
    train_dataset: SurvivalDataset,
    test_dataset: SurvivalDataset,
    best_features: list,
    best_params: dict,
    random_state: int,
    ds: str,
    early_stopping_rounds: int = 50,
    scale: float = 1.0,
    plot_path: str | None = None,
    n_models: int = 10,
    subset_fraction: tuple[float, float] = (0.8, 0.8),
    aggregation: str = "mean",
    replace: bool = True,
) -> tuple[float, float, float] | float:
    """
    Test the model on the given test dataset using bagging.

    :param train_dataset: dataset to train the model on.
    :param test_dataset: dataset to test the model on.
    :param best_features: list of best features.
    :param best_params: best parameters for the model.
    :param random_state: random state for the test.
    :param ds: dataset to use.
    :param early_stopping_rounds: early stopping rounds, defaults to 50.
    :param scale: scales the predictions, defaults to 1.0.
    :param plot_path: path to save SHAP plot, defaults to None.
    :param n_models: number of models in bag, defaults to 10.
    :param subset_fraction: subset fractions bootstrapping (event, censored), defaults to (0.8, 0.8).
    :param aggregation: aggregation method, defaults to "mean".
    :param replace: whether to sample with replacement, defaults to True.
    :return: c-index, censoring accuracy, and MAE observed or evaluation metric.
    """
    models = []
    train_dataset.X = train_dataset.X[best_features]
    train_dataset.split(test_size=0.1, random_state=random_state)
    censored_train = train_dataset.censored_train["CENSORED"]
    censored_dataset = train_dataset.X_train.loc[censored_train == 1]
    event_dataset = train_dataset.X_train.loc[censored_train == 0]
    np.random.seed(random_state)
    for _ in range(n_models):
        train_indices = bootstrap_split(
            censored_dataset, event_dataset, subset_fraction, replace
        )
        train_dataset.split_on_indices((train_indices, None))
        clf = train_one_model(
            train_dataset,
            best_params,
            early_stopping_rounds=early_stopping_rounds,
        )
        models.append(clf)
    dtest = convert_to_dmatrix(
        test_dataset.X[best_features],
        test_dataset.y_lower_bound,
        test_dataset.y_upper_bound,
    )
    dtrain = convert_to_dmatrix(
        train_dataset.X[best_features],
        train_dataset.y_lower_bound,
        train_dataset.y_upper_bound,
    )
    ys = np.array([m.predict(dtest) * scale for m in models])
    ys_train = np.array([m.predict(dtrain) * scale for m in models])
    with open(os.path.join(plot_path, f"ys_train_{ds}.npy"), "wb") as f:
        np.save(f, ys_train)
    with open(os.path.join(plot_path, f"ys_test_{ds}.npy"), "wb") as f:
        np.save(f, ys)
    if aggregation == "mean":
        mean = np.mean(ys, axis=0)
        stdev = np.std(ys, axis=0) * 1.96
        ypred, ci_lower, ci_upper = (mean, mean - stdev, mean + stdev)
    if aggregation == "median":
        median = np.percentile(ys, axis=0, q=50)
        ci_lower = np.percentile(ys, axis=0, q=2.5)
        ci_upper = np.percentile(ys, axis=0, q=97.5)
        ypred, ci_lower, ci_upper = (median, ci_lower, ci_upper)

    cindex, censoring_acc, mae_observed = c_index(
        test_dataset.y_lower_bound,
        test_dataset.y_upper_bound,
        ypred,  # type: ignore
    )
    if ds == "full":
        plot_shap(models, test_dataset, best_features, plot_path=plot_path)
        plot_survival_time(test_dataset, ys, interval=True, plot_path=plot_path)
    return cindex, censoring_acc, mae_observed


def cross_validate_model(
    dataset: SurvivalDataset,
    xgbparams: dict,
    k_fold_seed: int,
    n_folds: int = 5,
    plot: bool = False,
    use_mcc: bool = False,
    early_stopping_rounds: int = 50,
) -> float:
    """
    Cross validate the model on the given dataset.

    :param dataset: dataset to cross validate the model on.
    :param xgbparams: parameters for the model.
    :param k_fold_seed: seed for the k-fold split.
    :param n_folds: number of folds, defaults to 5.
    :param plot: whether to plot the training, defaults to False.
    :param use_mcc: whether to use MCC instead of C-index as evaluation metric, defaults to False.
    :param early_stopping_rounds: early stopping rounds, defaults to 50.
    :return: mean evaluation metric.
    """
    results = []
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=k_fold_seed)
    for train_index, valid_index in folds.split(dataset.X, dataset.censored):
        fold_dataset = dataset.model_copy()
        fold_dataset.split_on_indices((train_index, valid_index))
        res = train_one_model(
            fold_dataset,
            xgbparams,
            plot,
            eval_metric="mcc" if use_mcc else "all",
            early_stopping_rounds=early_stopping_rounds,
        )

        results.append(res)
    return results


def cross_validate_model_bagging(
    dataset: SurvivalDataset,
    xgbparams: dict,
    k_fold_seed: int,
    n_folds: int = 5,
    plot: bool = False,
    use_mcc: bool = False,
    early_stopping_rounds: int = 50,
    n_models: int = 10,
    subset_fraction: float = 0.8,
    aggregation: str = "mean",
    replace: bool = True,
) -> float:
    """
    Cross validate the model on the given dataset using bagging.

    :param dataset: dataset to cross validate the model on.
    :param xgbparams: parameters for the model.
    :param k_fold_seed: seed for the k-fold split.
    :param n_folds: number of folds, defaults to 5.
    :param n_models: _description_, defaults to 10
    :param subset_fraction: _description_, defaults to 0.8
    :param aggregation: _description_, defaults to "mean"
    :param replace: _description_, defaults to True
    :param plot: whether to plot the training, defaults to False.
    :param use_mcc: whether to use MCC instead of C-index as evaluation metric, defaults to False.
    :param early_stopping_rounds: early stopping rounds, defaults to 50.
    :return: mean evaluation metric.
    """
    results = []
    eval_metric = "mcc" if use_mcc else "all"
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=k_fold_seed)
    for train_index, valid_index in folds.split(dataset.X, dataset.censored):
        models = []

        fold_dataset = dataset.model_copy()
        fold_dataset.split_on_indices((train_index, valid_index))

        dvalid = convert_to_dmatrix(
            fold_dataset.X_val,
            fold_dataset.y_lower_bound_val,
            fold_dataset.y_upper_bound_val,
        )
        censored_train = fold_dataset.censored_train["CENSORED"]
        censored_dataset = fold_dataset.X_train.loc[censored_train == 1]
        event_dataset = fold_dataset.X_train.loc[censored_train == 0]
        np.random.seed(k_fold_seed)
        for _ in range(n_models):
            train_indices = bootstrap_split(
                censored_dataset, event_dataset, subset_fraction, replace
            )
            train_indices, valid_indices = train_test_split(
                train_indices,
                test_size=0.2,
                random_state=42,
                stratify=fold_dataset.censored.iloc[train_indices],
            )

            m_fold_dataset = fold_dataset.model_copy()
            m_fold_dataset.split_on_indices((train_indices, valid_indices))
            clf = train_one_model(
                m_fold_dataset,
                xgbparams,
                plot=plot,
                early_stopping_rounds=early_stopping_rounds,
            )
            models.append(clf)
        ys = np.array([m.predict(dvalid) for m in models])
        if aggregation == "mean":
            mean = np.mean(ys, axis=0)
            stdev = np.std(ys, axis=0) * 1.96
            ypred, ci_lower, ci_upper = (mean, mean - stdev, mean + stdev)
        if aggregation == "median":
            median = np.percentile(ys, axis=0, q=50)
            ci_lower = np.percentile(ys, axis=0, q=2.5)
            ci_upper = np.percentile(ys, axis=0, q=97.5)
            ypred, ci_lower, ci_upper = (median, ci_lower, ci_upper)

        cindex, censoring_acc, mae_observed = c_index(
            fold_dataset.y_lower_bound_val,
            fold_dataset.y_upper_bound_val,
            ypred,
        )
        if eval_metric == "cindex":
            res = cindex
        elif eval_metric == "mcc":
            res = calculate_mcc(cindex, censoring_acc, mae_observed)
        else:
            res = cindex, censoring_acc, mae_observed
        results.append(res)
    return results


def cross_validate_model_scale_search(
    dataset: SurvivalDataset,
    xgbparams: dict,
    k_fold_seed: int,
    n_folds: int = 5,
    early_stopping_rounds: int = 50,
) -> float:
    """
    Cross validate the model on the given dataset to find the best scale.

    :param dataset: dataset to cross validate the model on.
    :param xgbparams: parameters for the model.
    :param k_fold_seed: seed for the k-fold split.
    :param n_folds: number of folds, defaults to 5.
    :param early_stopping_rounds: early stopping rounds, defaults to 50.
    :return: the best scale.
    """
    best_scales = []
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=k_fold_seed)
    for train_index, valid_index in folds.split(dataset.X, dataset.censored):
        fold_dataset = dataset.model_copy()
        fold_dataset.split_on_indices((train_index, valid_index))
        clf = train_one_model(
            fold_dataset,
            xgbparams,
            early_stopping_rounds=early_stopping_rounds,
        )
        dvalid = convert_to_dmatrix(
            fold_dataset.X_val,
            fold_dataset.y_lower_bound_val,
            fold_dataset.y_upper_bound_val,
        )
        pred = clf.predict(dvalid)
        res = []
        for i in np.linspace(0.01, 1, 100):
            new_pred = pred * i
            cindex, censoring_acc, mae_observed = c_index(
                fold_dataset.y_lower_bound_val,
                fold_dataset.y_upper_bound_val,
                new_pred,
            )
            event_observed = (
                fold_dataset.y_lower_bound_val == fold_dataset.y_upper_bound_val
            ).astype(int)
            mae_censored = mean_absolute_error(
                fold_dataset.y_lower_bound_val[event_observed == 0],
                new_pred[event_observed == 0],
            )
            res.append((mae_observed, censoring_acc, mae_censored, cindex, i))
        best_scales.append(min(res, key=lambda x: x[0])[4])

    return np.mean(best_scales)


def tune_hp(
    dataset: SurvivalDataset,
    use_mcc: bool = False,
) -> dict:
    """
    Tune the hyperparameters of the model using cross-validation.

    :param dataset: dataset to tune the hyperparameters on.
    :param use_mcc: whether to use MCC as metric, defaults to False.
    :return: dictionary of results.
    """
    param_grid = {
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 5],
        "subsample": [1.0],
        "colsample_bynode": [1.0],
        "objective": ["survival:aft"],
        "eval_metric": ["aft-nloglik"],
        "aft_loss_distribution": ["normal", "extreme"],
        "aft_loss_distribution_scale": [0.1],
        "tree_method": ["hist"],
        "booster": ["gbtree"],
        "grow_policy": ["lossguide"],
        "lambda": [0.01],
        "alpha": [0.02],
        "n_jobs": [-1],
    }
    relevant_params = [
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "aft_loss_distribution",
        "aft_loss_distribution_scale",
        "alpha",
        "lambda",
    ]
    results = {}
    combinations = list(itertools.product(*param_grid.values()))
    for params in tqdm(combinations, total=len(combinations)):
        xgbparams = dict(zip(param_grid.keys(), params))
        result = cross_validate_model(
            dataset,
            xgbparams,
            k_fold_seed=0,
            use_mcc=use_mcc,
            early_stopping_rounds=50,
        )
        params_to_save = {k: xgbparams[k] for k in relevant_params}
        results[json.dumps(params_to_save)] = result
        with open("current_hp.json", "w") as f:
            json.dump(results, f)
    return results


def tune_hp_bagging(
    train_dataset: SurvivalDataset,
    best_params: dict,
    use_mcc: bool = False,
) -> dict:
    """
    Tune the hyperparameters of the model using cross-validation with bagging.

    :param train_dataset: dataset to tune the hyperparameters on.
    :param best_params: best parameters for the model.
    :param use_mcc: whether to use MCC as metric, defaults to False.
    :return: dictionary of results.
    """
    param_grid = {
        "n_models": [25, 50, 75, 100],
        "subset_fraction": [
            (0.6, 0.6),
            (0.6, 0.8),
            (0.8, 0.6),
            (0.8, 0.8),
            (1.0, 0.6),
            (0.6, 1.0),
            (1.0, 0.8),
            (0.8, 1.0),
            (1.0, 1.0),
        ],
        "aggregation": ["mean", "median"],
        "replace": [True, False],
    }
    relevant_params = [
        "n_models",
        "subset_fraction",
        "aggregation",
        "replace",
    ]
    results = {}
    combinations = list(itertools.product(*param_grid.values()))
    for params in tqdm(combinations, total=len(combinations)):
        bagging_params = dict(zip(param_grid.keys(), params))
        result = cross_validate_model_bagging(
            dataset=train_dataset,
            xgbparams=best_params,
            k_fold_seed=0,
            early_stopping_rounds=50,
            use_mcc=use_mcc,
            **bagging_params,
        )
        params_to_save = {k: bagging_params[k] for k in relevant_params}
        results[json.dumps(params_to_save)] = result
        with open("current_hp.json", "w") as f:
            json.dump(results, f)
    return results
