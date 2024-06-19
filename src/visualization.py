import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

from survival_dataset import SurvivalDataset
from utils import c_index


def plot_feature_selectors(feature_selectors: dict, name: str, SAVE_PATH: str) -> None:
    """
    Plot the c-index of the feature selection algorithms.

    :param feature_selectors: feature selection results
    :param name: name of the dataset
    :param SAVE_PATH: path to save the plot
    """
    for selector, results in feature_selectors.items():
        plt.plot(
            range(
                len(results["mean_cindex"]),
                results["mean_cindex"],
            ),
            label=selector.title(),
        )
    plt.xlabel("# dimensions")
    plt.legend()
    plt.ylabel("cindex")
    plt.xlim(0, 100)
    plt.title(f"{name.capitalize()} Survival Prediction")
    plt.savefig(f"{SAVE_PATH}/{name}_features.pdf")
    plt.close()


def plot_shap(
    clf: BaseEstimator,
    train_dataset: SurvivalDataset,
    best_features: list,
    plot_path: str | None = None,
) -> None:
    """
    Calculate and plot SHAP values for the best features.

    :param clf: trained model.
    :param train_dataset: training dataset.
    :param best_features: list of best features.
    :param plot_path: path to save to, defaults to None
    """
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(train_dataset.X_train[best_features])

    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=14)
    plt.grid(axis="x", linestyle="--", color="grey", alpha=0.7)
    plt.xticks(fontsize=8)
    # rename the features with more than 4 decimals
    feature_names = [
        f"{float(f):.4f}" if len(str(f)) > 14 else f for f in best_features
    ]
    shap.summary_plot(
        shap_values,
        train_dataset.X_train[best_features],
        feature_names=feature_names,
        color_bar=True,
        show=False,
    )
    if plot_path:
        plt.savefig(os.path.join(plot_path, "shap_plot.pdf"))
    else:
        plt.show()
    plt.cla()
    plt.clf()


def plot_scale_search(
    y_lower_bound_val: np.array, y_upper_bound_val: np.ndarray, pred: np.ndarray
) -> None:
    """
    Plot the scale vs MAE observed, censoring accuracy, and C-index.

    :param y_lower_bound_val: lower bound of the validation set.
    :param y_upper_bound_val: upper bound of the validation set.
    :param pred: predicted values.
    """
    res = []
    from sklearn.metrics import mean_absolute_error

    for i in np.linspace(0.01, 1, 100):
        new_pred = pred * i
        cindex, censoring_acc, mae_observed = c_index(
            y_lower_bound_val,
            y_upper_bound_val,
            new_pred,
        )
        event_observed = (y_lower_bound_val == y_upper_bound_val).astype(int)
        mae_censored = mean_absolute_error(
            y_lower_bound_val[event_observed == 0], new_pred[event_observed == 0]
        )
        res.append((mae_observed, censoring_acc, mae_censored, cindex, i))
    best_scale = min(res, key=lambda x: x[0])[4]
    plt.plot([r[4] for r in res], [r[0] for r in res], label="MAE censored")
    plt.plot([r[4] for r in res], [r[1] for r in res], label="Censoring accuracy")
    plt.plot([r[4] for r in res], [r[3] for r in res], label="C-index")
    plt.legend()
    plt.axvline(best_scale, color="red", linestyle="--", label="Best scale")
    plt.xlabel("Scale")
    plt.ylabel("MAE observed")
    plt.title("Optimising the scale of the predictions")
    plt.show()


def plot_training(evals_result: dict) -> None:
    """
    Plot the training and validation log loss for an XGBoost model.

    :param evals_result: dictionary containing the training and validation log loss.
    """
    epochs = len(evals_result["train"]["aft-nloglik"])
    x_axis = range(0, epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, evals_result["train"]["aft-nloglik"], label="Train")
    plt.plot(x_axis, evals_result["valid"]["aft-nloglik"], label="Valid")
    plt.legend()
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss")
    plt.show()


def plot_boxplots(experiment_name: str, with_annotations: bool = False) -> None:
    """
    Plot boxplots of the C-index for different datasets.

    :param experiment_name: name of the experiment to load.
    :param with_annotations: whether to show annotations with mean, defaults to False.
    """
    all_results = {}
    for f in os.listdir(experiment_name):
        if f.endswith(".csv"):
            dataset = f.split(".csv")[0].split("test_results_")[-1].title()
            all_results[dataset] = pd.read_csv(f"{experiment_name}/{f}")
            all_results[dataset]["Dataset"] = dataset

    for d in all_results:
        print(f"Final results {d}:")
        df_dataset = all_results[d].drop(columns=["Dataset", "Unnamed: 0"])
        means = df_dataset.mean()
        stds = df_dataset.std()
        for col in df_dataset.columns:
            print(f"{col}: {means[col]} ± {stds[col]}")
        print("----------------")

    all_results = pd.concat(list(all_results.values()))

    plt.figure(figsize=(6, 6))
    sns.boxplot(
        data=all_results,
        x="Dataset",
        y="C-index",
        palette="pastel",
        linewidth=1.5,
        order=["Nmr", "Clinical", "Full"],
    )

    if with_annotations:
        medians = all_results.groupby("Dataset")["C-index"].median()
        for i, dataset in enumerate(["Nmr", "Clinical", "Full"]):
            median_val = medians[dataset]
            plt.text(
                i,
                median_val - 0.0025,
                f"{median_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

    plt.title("C-index Comparison Between Datasets", fontsize=14)
    plt.xlabel("Dataset Type", fontsize=12)
    plt.ylabel("C-index", fontsize=12)
    plt.grid(axis="y", linestyle="--", color="grey", alpha=0.7)
    plt.ylim(0.8, 0.86)
    plt.yticks(np.arange(0.8, 0.87, 0.01))
    plt.tight_layout()
    plt.xticks([0, 1, 2], ["NMR", "Clinical", "Clinical + NMR"])
    plt.savefig(os.path.join(experiment_name, "boxplot.pdf"))


def plot_glucose_levels(df: pd.DataFrame) -> None:
    """
    Plot the glucose levels for censored and event participants.

    :param df: dataframe containing the glucose levels.
    """
    plt.grid(axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.75)
    sns.set_palette("Set2")
    sns.histplot(
        df[df["CENSORED"] == 1]["GLUC_2"],
        bins=10,
        label="Censored",
        alpha=1,
        edgecolor=(0, 0, 0, 0.5),
    )
    sns.histplot(
        df[df["CENSORED"] == 0]["GLUC_2"],
        bins=10,
        label="Event",
        alpha=1,
        edgecolor=(0, 0, 0, 0.5),
    )
    sns.despine()

    plt.gca().set_xticks(np.arange(2, 9, 1))
    plt.gca().set_xlabel("Fasting glucose (mmol/L)")
    plt.gca().set_ylabel("Number of participants")
    plt.gca().set_title("Fasting glucose levels for censored and event participants")
    plt.tight_layout()
    plt.legend()
    plt.savefig("../../methods/glucose_levels.pdf")


def plot_one_spectrum(df: pd.DataFrame) -> None:
    """
    Plot a single NMR spectrum.

    :param df: df containing the NMR data.
    """
    unique_participants = df["ID"].unique().tolist()
    one_df = df[df["ID"] == unique_participants[0]]
    plt.plot(one_df["Chemical Shifts"], one_df["PPM"], linestyle="-", color="black")
    sns.despine()
    sns.set_style("white")
    plt.xlabel("PPM")
    plt.ylabel("Signal Intensity")
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.savefig("../../methods/nmr_spectrum.pdf")


def plot_survival(df: pd.DataFrame) -> None:
    """
    Plot the survival times for censoring and event participants.

    :param df: dataframe containing the survival times.
    """
    plt.grid(axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.75)
    sns.set_palette("Set2")
    sns.histplot(
        df[df["CENSORED"] == 1]["lower_bound"],
        bins=10,
        label="Censored",
        alpha=1,
        edgecolor=(0, 0, 0, 0.5),
    )
    sns.histplot(
        df[df["CENSORED"] == 0]["lower_bound"],
        bins=10,
        label="Event",
        alpha=1,
        edgecolor=(0, 0, 0, 0.5),
    )
    plt.gca().set_xticks(np.arange(0, 11, 1))
    plt.gca().set_xlabel("Survival Time in Years")
    plt.gca().set_ylabel("Number of Participants")
    plt.gca().set_title("Survival time for censored and event participants")
    plt.tight_layout()
    plt.legend()
    plt.savefig("../../methods/survival_times.pdf")


def plot_cumulative_variance() -> None:
    """Plot the cumulative variance explained by the principal components."""
    X_train_nmr = pd.read_csv("../../datasets.nosync/nmr_train.csv")
    X_train_nmr = X_train_nmr.drop(columns=["CENSORED", "upper_bound", "lower_bound"])
    pca_nmr = PCA()
    pca_nmr.fit(X_train_nmr)
    cumulative_variance_nmr = np.cumsum(pca_nmr.explained_variance_ratio_)

    X_train_clinical = pd.read_csv("../../datasets.nosync/clinical_train.csv")
    X_train_clinical = X_train_clinical.drop(
        columns=["CENSORED", "upper_bound", "lower_bound"]
    )
    X_train_clinical = X_train_clinical.fillna(X_train_clinical.median())
    pca_clinical = PCA()
    pca_clinical.fit(X_train_clinical)
    cumulative_variance_clinical = np.cumsum(pca_clinical.explained_variance_ratio_)

    sns.set_palette("pastel")
    sns.lineplot(
        cumulative_variance_nmr[:100] * 100,
        marker="v",
        linestyle="-",
        alpha=1,
        label="NMR",
    )
    sns.lineplot(
        cumulative_variance_clinical[:100] * 100,
        marker="v",
        linestyle="-",
        alpha=1,
        label="Clinical",
    )

    plt.xlabel("Number of Principal Components (PCs)")
    plt.ylabel("Cumulative variance (%)")
    plt.title("Cumulative variance explained by PCs")
    plt.legend(loc="lower right")
    plt.xlim(0, 30)
    plt.grid()
    plt.savefig("../../methods/cumulative_variance.pdf")
