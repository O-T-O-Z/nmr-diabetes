import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from runner_functions import (
    SurvivalDataset,
    cross_validate_model,
    cross_validate_model_bagging,
    cross_validate_model_scale_search,
    test_model,
    test_model_bagging,
    tune_hp,
    tune_hp_bagging,
)
from utils import calculate_mcc, get_best_features_and_params, load_best_features
from visualization import plot_boxplots


def perform_search(
    dataset: SurvivalDataset,
    feature_selection_path: str,
    params_path: str,
    use_mcc: bool = False,
    bagging: bool = False,
    best_params: dict | None = None,
) -> tuple[dict, list]:
    """
    Perform hyperparameter search for the given dataset and feature selection.

    :param dataset: dataset to perform search on.
    :param feature_selection_path: path to the feature selection results.
    :param params_path: path to load or save the search results.
    :param use_mcc: whether to use MCC as metric, defaults to False.
    :param bagging: whether bagging hyperparams are being searched, defaults to False.
    :param best_params: best parameters to use during bagging, defaults to {}.
    :return: tuple of best parameters and best features.
    """
    subset = load_best_features(feature_selection_path)
    if os.path.exists(params_path) and not bagging:
        print("Loading previous search results", params_path)
        return get_best_features_and_params(feature_selection_path, params_path)
    if os.path.exists(params_path) and bagging:
        subset = load_best_features(feature_selection_path)
        with open(params_path, "r") as f:
            results = json.load(f)
        best_params = json.loads(max(results, key=results.get))
        return best_params, subset
    print("Performing search", feature_selection_path)
    dataset.X = dataset.X[subset]
    if bagging:
        results = tune_hp_bagging(
            dataset,
            best_params=best_params if bagging else {},
            use_mcc=use_mcc,
        )
    else:
        results = tune_hp(dataset, use_mcc=use_mcc)
    with open(params_path, "w") as f:
        json.dump(results, f)
    if bagging:
        return json.loads(max(results, key=results.get)), subset
    return get_best_features_and_params(feature_selection_path, params_path)


def perform_final_cross_validation(
    dataset: SurvivalDataset,
    best_features: list,
    best_params: dict,
    use_mcc: bool = False,
    bagging: bool = False,
    bagging_params: dict | None = None,
) -> None:
    """
    Perform final cross-validation on the dataset.

    :param dataset: dataset to perform cross-validation on.
    :param best_features: best features to use.
    :param best_params: best parameters to use.
    :param use_mcc: whether to use MCC as metric, defaults to False.
    :param bagging: whether bagging hyperparams are being searched, defaults to False.
    :param bagging_params: parameters to use during bagging, defaults to {}.
    """
    dataset.X = dataset.X[best_features]
    all_results = []
    func = cross_validate_model_bagging if bagging else cross_validate_model
    kwargs = bagging_params if bagging else {}
    print("Cross validating...")
    for random_state in tqdm([42, 55, 875]):
        results = func(
            dataset,
            best_params,
            random_state,
            n_folds=5,
            early_stopping_rounds=50,
            use_mcc=use_mcc,
            **kwargs,
        )
        all_results.append(results)
    print("Mean c-index:", np.mean(all_results))
    print("Std c-index:", np.std(all_results))


def perform_test(
    train_dataset: SurvivalDataset,
    test_dataset: SurvivalDataset,
    best_features: list,
    best_params: dict,
    ds: str,
    scale: float = 1.0,
    plot_path: str = None,
    bagging: bool = False,
    bagging_params: dict | None = None,
) -> pd.DataFrame:
    """
    Test the model on the test dataset.

    :param train_dataset: training dataset.
    :param test_dataset: test dataset.
    :param best_features: best features to use.
    :param best_params: best parameters to use.
    :param ds: dataset name.
    :param scale: scale to scale the predictions by, defaults to 1.0.
    :param plot_path: path to store plots, defaults to None.
    :param bagging: whether bagging is tested, defaults to False.
    :param bagging_params: bagging parameters, defaults to {}.
    :return: dataframe containing the results.
    """
    print("------------")
    all_results = pd.DataFrame(
        columns=["C-index", "Censoring accuracy", "MAE observed"]
    )
    func = test_model_bagging if bagging else test_model
    kwargs = bagging_params if bagging else {}

    for random_state in [84, 110, 1750, 2024, 7041]:
        cindex, censoring_acc, mae_observed = func(
            train_dataset,
            test_dataset,
            best_features,
            best_params,
            random_state,
            ds=ds,
            early_stopping_rounds=50,
            scale=scale,
            plot_path=plot_path,
            **kwargs,
        )

        all_results.loc[random_state] = [cindex, censoring_acc, mae_observed]

    print("Final results:")
    means = all_results.mean()
    stds = all_results.std()
    for col in all_results.columns:
        print(f"{col}: {means[col]:.2f} ± {stds[col]:.2f}")
    return all_results


def save_table(save_path: str) -> None:
    """
    Save the results table to a latex file.

    :param save_path: path to save the results.
    """
    all_df = pd.DataFrame(
        columns=[
            "C-index",
            "\text{Acc}_{\text{censored}}",
            "\text{MAE}_{\text{event}}",
            "MCC",
        ],
        index=["nmr", "clinical", "full"],
    )

    for f in os.listdir(save_path):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(save_path, f))
            ds = f.split("_")[-1].split(".csv")[0]
            mcc = calculate_mcc(
                df["C-index"], df["Censoring accuracy"], df["MAE observed"]
            )
            mcc = f"{round(mcc.mean(), 3)} ± {round(mcc.std(), 3)}"
            c_index = (
                f"{round(df['C-index'].mean(), 3)} ± {round(df['C-index'].std(), 3)}"
            )
            cens_acc = f"{round(df['Censoring accuracy'].mean(), 3)} ± \
                {round(df['Censoring accuracy'].std(), 3)}"
            mae = f"{round(df['MAE observed'].mean(), 3)} ± {round(df['MAE observed'].std(), 3)}"

            all_df.loc[ds, "C-index"] = c_index
            all_df.loc[ds, "\text{Acc}_{\text{censored}}"] = cens_acc
            all_df.loc[ds, "\text{MAE}_{\text{event}}"] = mae
            all_df.loc[ds, "MCC"] = mcc

    # rename indices
    all_df.rename(
        index={"nmr": "NMR", "clinical": "Clinical", "full": "Clinical + NMR"},
        inplace=True,
    )
    all_df.to_latex(
        os.path.join(save_path, "results_table.tex"),
        multirow=True,
        bold_rows=True,
        caption="XGB-AFT model performance on each dataset.",
        label="tab:results_ex2",
        float_format="%.3f",
    )


def main() -> None:
    """
    Run experiments on each dataset.

    Steps:
    - Performing hyperparameter search
    - Performing cross-validation
    - Performing testing
    - Saving the results
    """
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--save", type=str, help="Path to save the results", required=True
    )
    parser.add_argument("--load-params", type=str, help="Path to load the params")
    parser.add_argument("--load-features", type=str, help="Path to load the features")
    parser.add_argument("--bagging", action="store_true", help="Use bagging")
    parser.add_argument(
        "--use-mcc", action="store_true", help="Use MCC metric for cross-validation"
    )

    args = parser.parse_args()
    save_path = os.path.join(
        "experiments", str(args.save) + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    save_path = "experiments/bagging2024-06-26_11-54-15"

    os.makedirs(save_path, exist_ok=True)
    datasets = [
        ("clinical", "clinical_train.csv"),
        ("nmr", "nmr_train.csv"),
        ("full", "full_train.csv"),
    ]

    for d, f in datasets:
        print(f)
        dataset = SurvivalDataset()
        dataset.load_data(f)
        dataset_test = SurvivalDataset()
        dataset_test.load_data(f"{d}_test.csv")

        features_path = f"../results_fs/{d}_{args.load_features}.txt"
        best_params, best_features = perform_search(
            dataset,
            features_path,
            save_path + f"/params_{d}.json",
            args.use_mcc,
        )
        dataset.X = dataset.X[best_features]
        if args.bagging:
            print("Performing bagging search")
            bagging_params, best_features = perform_search(
                dataset,
                features_path,
                save_path + f"/params_{d}_bagging.json",
                args.use_mcc,
                bagging=True,
                best_params=best_params,
            )
            print("Best bagging params:", bagging_params)
        else:
            bagging_params = {}
        if not args.bagging:
            scales = [
                cross_validate_model_scale_search(
                    dataset,
                    best_params,
                    k_fold_seed=random_state,
                )
                for random_state in [42, 55, 875]
            ]
            best_scale = np.mean(scales)
            print("Best scale:", best_scale)
        perform_final_cross_validation(
            dataset,
            best_features,
            best_params,
            args.use_mcc,
            args.bagging,
            bagging_params,
        )
        all_results = perform_test(
            dataset,
            dataset_test,
            best_features,
            best_params,
            d,
            1.0,
            save_path,
            args.bagging,
            bagging_params,
        )
        all_results.to_csv(f"{save_path}/test_results_{d}.csv")
    save_table(save_path)
    plot_boxplots(save_path)


if __name__ == "__main__":
    main()
