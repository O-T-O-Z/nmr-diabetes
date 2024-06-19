import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

from src.runner_functions import SurvivalDataset
from utils import load_best_features


class CoxPHWrapper(BaseEstimator, RegressorMixin):
    """CoxPHWrapper that wraps the CoxPHFitter from lifelines to be used in sklearn pipelines."""

    def __init__(
        self, penalizer: float = 0.0, alpha: float = 0.05, l1_ratio: float = 0.0
    ) -> None:
        """
        Initialize the CoxPHWrapper.

        :param penalizer: coefficient penalty, defaults to 0.0
        :param alpha: confidence interval level, defaults to 0.05
        :param l1_ratio: L1 vs L2 ratio, defaults to 0.0
        """
        self.penalizer = penalizer
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = CoxPHFitter(
            penalizer=self.penalizer, alpha=self.alpha, l1_ratio=self.l1_ratio
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> BaseEstimator:
        """
        Fit the model to the dataframe by creating a full one.

        :param X: inputs dataframe.
        :param y: targets dataframe.
        :return: fitted model.
        """
        df = X.copy()
        df["survival"] = y["survival"]
        df["event"] = y["event"]
        self.model.fit(df, duration_col="survival", event_col="event", **kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Predict the survival function.

        :param X: inputs dataframe.
        :return: predicted survival function.
        """
        return self.model.predict_expectation(X)

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Score the model using the concordance index.

        :param X: inputs dataframe.
        :param y: targets dataframe.
        :return: concordance index.
        """
        test_df = X.copy()
        test_df["survival"] = y["survival"]
        test_df["event"] = y["event"]
        return self.model.score(test_df, scoring_method="concordance_index")


def score_model(y_true: np.array, y_pred: np.array, risk: bool = False) -> float:
    """
    Score the model using the concordance index.

    :param y_true: true values.
    :param y_pred: predicted values.
    :param risk: whether to use the risk score, defaults to False.
    :return: concordance index.
    """
    sign = 1 if risk else -1
    return concordance_index(y_true["survival"], y_pred * sign, y_true["event"])


def create_df_and_array(
    dataset: SurvivalDataset, best_features: list
) -> tuple[pd.DataFrame, np.array, np.array]:
    """
    Create a DataFrame and numpy array from the dataset.

    :param dataset: dataset to create the DataFrame and numpy array from.
    :param best_features: best features to use.
    """
    data_x = dataset.X_train[best_features]
    data_y_e = (dataset.y_lower_bound_train == dataset.y_upper_bound_train).values
    data_y_t = dataset.y_lower_bound_train.values
    data_y = np.array(
        list(zip(data_y_e, data_y_t)),
        dtype=[("event", "?"), ("survival", "<f8")],
    )
    df = pd.DataFrame(data_x, columns=best_features)
    df["survival"] = data_y_t
    df["event"] = data_y_e
    return df, data_x, data_y


def load_datasets(
    ds_name: str,
) -> tuple[pd.DataFrame, np.array, np.array, pd.DataFrame, np.array, np.array]:
    """
    Load the dataset in two ways, both a DataFrame and numpy array.

    :param ds_name: name of the dataset to load.
    :return: DataFrame and numpy array of the dataset.
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

    return (
        create_df_and_array(train_dataset, best_features),
        create_df_and_array(test_dataset, best_features),
    )


def fit_coxph(
    train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int
) -> float:
    """
    Fit the CoxPH model by performing grid search first.

    :param train_df: training DataFrame.
    :param test_df: testing DataFrame.
    :param random_state: random state to use.
    :return: concordance index.
    """
    param_grid = {
        "penalizer": [0.01, 0.1, 1.0],
        "l1_ratio": [0.0, 0.1, 0.5, 1.0],
        "alpha": [0.05, 0.1, 0.2, 0.5],
    }
    coxph = CoxPHWrapper()
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=coxph,
        param_grid=param_grid,
        scoring=make_scorer(score_model, greater_is_better=True),
        cv=kf,
        n_jobs=-1,
    )
    grid_search.fit(
        train_df.drop(columns=["survival", "event"]),
        train_df[["survival", "event"]],
        fit_options={"step_size": 0.5},
    )

    coxph = CoxPHFitter(
        penalizer=grid_search.best_params_["penalizer"],
        l1_ratio=grid_search.best_params_["l1_ratio"],
        alpha=grid_search.best_params_["alpha"],
    )
    coxph.fit(
        train_df,
        duration_col="survival",
        event_col="event",
        fit_options={"step_size": 0.5},
    )

    return coxph.score(test_df, scoring_method="concordance_index")


def fit_survival_trees(
    train_data_x: np.array,
    train_data_y: np.array,
    test_data_x: np.array,
    test_data_y: np.array,
    random_state: int,
) -> dict[str, float]:
    """
    Fit the survival tree models by performing grid search first.

    :param train_data_x: input training data.
    :param train_data_y: target training data.
    :param test_data_x: input testing data.
    :param test_data_y: target testing data.
    :param random_state: random state to use.
    :return: concordance indices.
    """
    models = {
        "RSF": RandomSurvivalForest,
        "GBS": GradientBoostingSurvivalAnalysis,
    }
    results = {}
    for name, model in models.items():
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(
            estimator=model(random_state=random_state),
            param_grid={
                "n_estimators": [10, 50, 100, 200],
                "max_depth": [2, 3, 4, 5],
            },
            scoring=make_scorer(score_model, greater_is_better=True, risk=True),
            cv=kf,
            n_jobs=-1,
        )
        grid_search.fit(train_data_x, train_data_y)
        model = model(random_state=random_state, **grid_search.best_params_)
        model.fit(train_data_x, train_data_y)
        pred_scores = model.predict(test_data_x)
        c_index = concordance_index(
            test_data_y["survival"], -pred_scores, test_data_y["event"]
        )
        results[name] = c_index
    return results


if __name__ == "__main__":
    for ds_name in ["full", "clinical", "nmr"]:
        df, data_x, data_y, test_df, test_data_x, test_data_y = load_datasets(ds_name)
        res = {"CoxPH": [], "RSF": [], "GBS": []}
        for random_state in [42, 55, 875]:
            res["CoxPH"].append(
                fit_coxph(
                    df,
                    test_df,
                    random_state=random_state,
                )
            )
            results_cox = fit_survival_trees(
                data_x, data_y, test_data_x, test_data_y, random_state=random_state
            )
            for key, value in results_cox.items():
                res[key].append(value)
        results_df = pd.DataFrame(res)
        print(results_df)
        results_df.to_csv(f"../results_models/baselines/{ds_name}_baseline_results.csv")
