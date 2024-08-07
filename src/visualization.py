import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

from survival_dataset import SurvivalDataset
from utils import c_index, convert_to_dmatrix


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
            ),
            results["mean_cindex"],
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
    clf: BaseEstimator | list[BaseEstimator],
    test_dataset: SurvivalDataset,
    best_features: list,
    plot_path: str | None = None,
) -> None:
    """
    Calculate and plot SHAP values for the best features.

    :param clf: trained model.
    :param test_dataset: training dataset.
    :param best_features: list of best features.
    :param plot_path: path to save to, defaults to None.
    """
    if isinstance(clf, list):
        shap_values_list = []
        for model in clf:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(test_dataset.X[best_features])
            shap_values_list.append(shap_values)
        stacked_shap_values = np.stack(shap_values_list, axis=0)
        shap_values = np.mean(stacked_shap_values, axis=0)
    else:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(test_dataset.X[best_features])

    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=14)
    plt.grid(axis="x", linestyle="--", color="grey", alpha=0.7)
    plt.xticks(fontsize=8)
    features_map = {
        "GLUC_2": "Glucose",
        "WAIST_2B": "Waist circumference",
        "ANTIHYP_2B": "Antihypertensive medication",
        "TGL_2": "Triglycerides",
        "Creat_NC2": "Creatinine",
        "V30_2B": "Years of smoking",
        "RACE": "Race",
        "CRP_2": "C-reactive protein",
    }
    # rename the features with more than 4 decimals
    feature_names = [
        f"{float(f):.4f} ppm" if len(str(f)) > 14 else features_map[f]
        for f in best_features
    ]

    shap.summary_plot(
        shap_values,
        test_dataset.X[best_features],
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


def plot_single_shap(
    clf: BaseEstimator,
    test_dataset: SurvivalDataset,
    best_features: list,
    is_censored: bool,
    is_correct: bool,
    plot_path: str | None = None,
) -> None:
    """
    Calculate and plot SHAP values for three individual participants.

    :param clf: trained model.
    :param test_dataset: training dataset.
    :param best_features: list of best features.
    :param is_censored: whether the participant is censored.
    :param is_correct: whether the prediction is correct.
    :param plot_path: path to save to, defaults to None.
    """
    shap_explanations = []
    features_map = {
        "GLUC_2": "Glucose",
        "WAIST_2B": "Waist circumference",
        "ANTIHYP_2B": "Antihypertensive medication",
        "TGL_2": "Triglycerides",
        "Creat_NC2": "Creatinine",
        "V30_2B": "Years of smoking",
        "RACE": "Race",
        "CRP_2": "C-reactive protein",
    }
    # rename the features with more than 4 decimals
    feature_names = [
        f"{float(f):.4f} ppm" if len(str(f)) > 14 else features_map[f]
        for f in best_features
    ]

    dtest = convert_to_dmatrix(
        test_dataset.X[best_features],
        test_dataset.y_lower_bound,
        test_dataset.y_upper_bound,
    )
    ys = np.array([m.predict(dtest) for m in clf])
    mean = np.mean(ys, axis=0)
    stdev = np.std(ys, axis=0) * 1.96
    for model in clf:
        explainer = shap.Explainer(
            model, test_dataset.X[best_features], feature_names=feature_names
        )
        shap_explanation = explainer(test_dataset.X[best_features])
        shap_explanations.append(shap_explanation)

    # Combine the explanations
    combined_shap_values = np.mean([exp.values for exp in shap_explanations], axis=0)
    combined_base_values = np.mean(
        [exp.base_values for exp in shap_explanations], axis=0
    )

    # Create a new Explanation object with the combined values
    final_explanation = shap.Explanation(
        values=combined_shap_values,
        base_values=combined_base_values,
        data=test_dataset.X[best_features].values,
        feature_names=feature_names,
    )

    corr = "correct" if is_correct else "incorrect"
    cens = "censored" if is_censored else "event"
    fig, axes = plt.subplots(3, 1, sharex=True)

    for i, ax in enumerate(axes):
        plt.sca(ax)
        ax.set_title(
            f"Ground truth: {test_dataset.y_lower_bound.iloc[i]:.2f} years, "
            f"Prediction: {mean[i]:.2f} years ± {stdev[i]:.2f} years"
        )
        ax.set_xticks([])
        ax.set_xlabel("")
        shap.waterfall_plot(
            final_explanation[i],
            show=False,
        )
    fig.suptitle(f"SHAP values for three {corr} {cens} participants", fontsize=16)
    fig.set_size_inches(12, 10)
    fig.tight_layout()
    if plot_path:
        plt.savefig(
            os.path.join(
                plot_path,
                f"shap_plot_{corr}_{cens}.pdf",
            )
        )
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
        if f.endswith(".csv") and "test_results" in f:
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
        showmeans=True,
        meanprops={
            "marker": "d",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
        },
        order=["Nmr", "Clinical", "Full"],
    )
    sns.despine()
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

    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("C-index", fontsize=12)
    plt.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    plt.tick_params(axis="y", which="both", left=True, right=False, labelbottom=True)
    plt.grid(axis="y", linestyle="--", color="grey", alpha=0.7)
    plt.ylim(0.8, 0.88)
    plt.yticks(np.arange(0.8, 0.89, 0.01))
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
        color=sns.color_palette("Set2")[0],
    )
    sns.histplot(
        df[df["CENSORED"] == 0]["GLUC_2"],
        bins=10,
        label="Event",
        alpha=1,
        edgecolor=(0, 0, 0, 0.5),
        color=sns.color_palette("Set2")[1],
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


def plot_multiple_spectra(nmr_df: pd.DataFrame) -> None:
    """
    Plot multiple NMR spectra.

    :param nmr_df: df containing the NMR data.
    """
    censored_nmr = nmr_df[nmr_df["CENSORED"] == 1]
    event_nmr = nmr_df[nmr_df["CENSORED"] == 0]
    censored_nmr = censored_nmr.drop(columns=["CENSORED", "upper_bound", "lower_bound"])
    event_nmr = event_nmr.drop(columns=["CENSORED", "upper_bound", "lower_bound"])

    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    sns.set_palette("Set2")
    # first two colors
    color1 = sns.color_palette("Set2")[0]
    color2 = sns.color_palette("Set2")[1]
    # plot 3 censored and 3 event participants
    for i in range(3):
        one_df_censored = censored_nmr.iloc[i + random.randint(0, 100)]
        one_df_event = event_nmr.iloc[i + random.randint(0, 100)]

        ax[0, i].plot(
            [float(i) for i in one_df_censored.index.values],
            [float(i) for i in one_df_censored.values],
            linestyle="-",
            color=color1,
        )
        ax[1, i].plot(
            [float(i) for i in one_df_event.index.values],
            [float(i) for i in one_df_event.values],
            linestyle="-",
            color=color2,
        )
        ax[0, i].invert_xaxis()
        ax[0, i].set_xticklabels([])
        ax[1, i].invert_xaxis()
        ax[1, i].tick_params(
            axis="x", which="both", bottom=True, top=False, labelbottom=True
        )
        ax[1, i].set_xlabel("PPM")

    legend_elements = [
        Line2D([0], [0], color=color1, lw=2, label="Censored"),
        Line2D([0], [0], color=color2, lw=2, label="Event"),
    ]
    ax[0, 2].legend(handles=legend_elements, loc="upper right")
    ax[0, 0].set_ylabel("Signal Intensity")
    ax[1, 0].set_ylabel("Signal Intensity")

    ax[0, 0].tick_params(
        axis="both",
        which="both",
        left=True,
        labelbottom=True,
        labelleft=True,
    )
    ax[1, 0].tick_params(
        axis="both",
        which="both",
        left=True,
        labelbottom=True,
        labelleft=True,
    )
    sns.despine()
    sns.set_style("white")
    fig.suptitle("NMR Spectra of Censored and Event Participants")
    # create a legend with labels censored and event with colors
    plt.tight_layout()
    plt.savefig("../methods/nmr_spectra_comparison.pdf")
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


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
    sns.despine()
    plt.gca().set_xticks(np.arange(0, 11, 1))
    plt.gca().set_xlabel("Survival Time in Years")
    plt.gca().set_ylabel("Number of Participants")
    plt.gca().set_title("Survival time for censored and event participants")
    plt.tight_layout()
    plt.legend()
    plt.savefig("../../methods/survival_times.pdf")


def plot_cumulative_variance() -> None:
    """Plot the cumulative variance explained by the principal components."""
    X_train_nmr = pd.read_csv("../datasets.nosync/nmr_train.csv")
    X_train_nmr = X_train_nmr.drop(columns=["CENSORED", "upper_bound", "lower_bound"])
    pca_nmr = PCA()
    pca_nmr.fit(X_train_nmr)
    cumulative_variance_nmr = np.cumsum(pca_nmr.explained_variance_ratio_)

    X_train_clinical = pd.read_csv("../datasets.nosync/clinical_train.csv")
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
    plt.savefig("../methods/cumulative_variance.pdf")


def plot_difference(nmr_df: pd.DataFrame, features: list | None = None) -> None:
    """
    Plot multiple NMR spectra.

    :param nmr_df: df containing the NMR data.
    :param features: list of features to highlight, defaults to None.
    """
    censored_nmr = nmr_df[nmr_df["CENSORED"] == 1]
    event_nmr = nmr_df[nmr_df["CENSORED"] == 0]
    censored_nmr = censored_nmr.drop(columns=["CENSORED", "upper_bound", "lower_bound"])
    event_nmr = event_nmr.drop(columns=["CENSORED", "upper_bound", "lower_bound"])

    event_df = event_nmr.sum(axis=0) / len(event_nmr)
    censored_df = censored_nmr.sum(axis=0) / len(censored_nmr)

    censored_diff = event_df - censored_df
    plt.plot(
        [float(idx) for idx in censored_diff.index],
        list(censored_diff.values),
        linestyle="-",
        label="Difference",
        color="black",
    )

    plt.xlabel("PPM")
    plt.ylabel("Signal Intensity")
    plt.title(
        "NMR Spectrum of the Difference between Censored and Event Data", fontsize=12
    )
    plt.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        labelbottom=True,
        left=True,
        labelleft=True,
    )
    if features:
        for f in features:
            plt.axvline(x=float(f), color="green", linestyle="--")
        plt.legend(["Difference", "Selected Features"])
        save_path = "../methods/nmr_difference_features.pdf"
    else:
        save_path = "../methods/nmr_difference.pdf"
    sns.despine()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


def plot_survival_time(
    test_dataset: SurvivalDataset,
    predicted_times: list[float] | list[list[float]],
    interval: bool = False,
    plot_path: str | None = None,
) -> None:
    """
    Plot the true vs predicted survival times.

    :param test_dataset: true survival times (dataset).
    :param predicted_times: predicted survival times.
    :param interval: whether to show a prediction interval, defaults to False.
    :param plot_path: path to save to, defaults to None
    """
    upper_bound = np.array(test_dataset.y_upper_bound)
    predicted_times = np.array(predicted_times)

    # get 2 censored and 3 event participants
    censored = 0
    event = 0
    selected = []

    for i, t in enumerate(upper_bound):
        if np.isinf(t) and censored < 2:
            selected.append(i)
            censored += 1
        elif not np.isinf(t) and event < 3:
            selected.append(i)
            event += 1
        if censored == 2 and event == 3:
            break

    true_times = test_dataset.y_upper_bound[selected]
    true_times_lower = test_dataset.y_lower_bound[selected]
    if isinstance(predicted_times[0], np.ndarray):
        predicted_times = predicted_times[:, selected]
    else:
        predicted_times = predicted_times[selected]

    data = {
        "patient": ["Patient X", "Patient E", "Patient D", "Patient N", "Patient I"],
        "actual_event": true_times,
        "predicted_event": predicted_times,
        "censored": [np.isinf(t) for t in true_times],
        "pred_error": "",
    }
    if isinstance(predicted_times[0], np.ndarray):
        data["predicted_event"] = data["predicted_event"].mean(axis=0)
        data["pred_error"] = data["predicted_event"].std(axis=0)
        interval = True
    df = pd.DataFrame(data).reset_index(drop=True)

    start_study = 0
    ind = df["predicted_event"].argmax()
    end_study = df.iloc[ind]["predicted_event"]
    if interval:
        end_study += df.iloc[ind]["pred_error"]
    sns.set_theme(style="whitegrid")
    sns.set_palette("Set2")
    sns_green, sns_orange, sns_blue = sns.color_palette("Set2")[:3]
    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(df["patient"])):
        if not df.loc[i, "censored"]:
            maximum = min(df.loc[i, "actual_event"], df.loc[i, "predicted_event"])
        else:
            maximum = max(true_times_lower[i], df.loc[i, "predicted_event"])
            if interval:
                maximum += df.loc[i, "pred_error"]

        ax.plot(
            [start_study, maximum],
            [i, i],
            "k-",
            linewidth=1,
        )

        if df.loc[i, "actual_event"] != np.inf:
            ax.plot(
                df.loc[i, "actual_event"],
                i,
                "X",
                markersize=10,
                markeredgewidth=2,
                color=sns_orange,
            )

            # Add arrow for ongoing events
            if df.loc[i, "actual_event"] < end_study:
                ax.annotate(
                    "",
                    xy=(end_study, i),
                    xytext=(df.loc[i, "actual_event"], i),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": sns_orange,
                        "linestyle": "--",
                    },
                    color=sns_orange,
                )
        else:
            ax.plot(
                true_times_lower[i],
                i,
                "X",
                markersize=10,
                markeredgewidth=2,
                color=sns_green,
                zorder=10,
            )
            ax.annotate(
                "",
                xy=(end_study, i),
                xytext=(df.loc[i, "predicted_event"], i),
                arrowprops={
                    "arrowstyle": "->",
                    "color": sns_blue,
                    "linestyle": "--",
                },
                color=sns_blue,
                zorder=10,
            )

        if interval:
            ax.errorbar(
                df.loc[i, "predicted_event"],
                i,
                xerr=df.loc[i, "pred_error"],
                fmt="X",
                markersize=10,
                markeredgewidth=2,
                capsize=5,
                color=sns_blue,
                zorder=10,
            )
        else:
            ax.plot(
                df.loc[i, "predicted_event"],
                i,
                "X",
                markersize=10,
                markeredgewidth=2,
                color=sns_blue,
                zorder=10,
            )

    ax.set_yticks(range(len(df["patient"])))
    ax.set_yticklabels(df["patient"])
    ax.set_xlabel("Survival time in years")
    ax.set_title("Patient Event Timeline: Actual vs Predicted")
    ax.plot([], [], "X", label="Censored")
    ax.plot([], [], "X", label="Event")
    ax.plot([], [], "X", label="Predicted event")
    ax.legend()
    ax.axvline(x=start_study, color="k", linestyle=":", linewidth=1)
    ax.text(start_study, -0.5, "Present", ha="center", va="top")
    ax.grid(False)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis="y", which="both", left=True, right=False, labelbottom=True)
    plt.tight_layout()
    sns.despine()
    sns.set_style("white")
    if not plot_path:
        plt.show()
    else:
        plt.savefig(os.path.join(plot_path, "patient_timeline.pdf"))
    ax.cla()
    plt.cla()
    plt.clf()
    plt.close()


def plot_survival_sd(
    test_dataset: SurvivalDataset,
    predicted_times: list[float],
    std_devs: list[float] | None = None,
    plot_path: str | None = None,
) -> None:
    """
    Plot the true vs predicted survival times with standard deviations.

    :param test_dataset: test dataset.
    :param predicted_times: predicted survival times.
    :param std_devs: standard deviations.
    :param plot_path: path to save to, defaults to None
    """
    sns_green, sns_orange = sns.color_palette("Set2")[:2]
    fig, ax = plt.subplots(figsize=(10, 6))

    if std_devs is None:
        std_devs = np.array([0] * len(test_dataset.y_lower_bound))
    censored_indices = [
        i
        for i in range(len(test_dataset.y_upper_bound))
        if np.isinf(test_dataset.y_upper_bound[i])
    ]
    true_times_event = test_dataset.y_lower_bound[
        [i for i in range(len(test_dataset.y_upper_bound)) if i not in censored_indices]
    ]
    true_times_censored = test_dataset.y_lower_bound[censored_indices]

    predicted_times_event = predicted_times[
        [i for i in range(len(test_dataset.y_upper_bound)) if i not in censored_indices]
    ]
    predicted_times_censored = predicted_times[censored_indices]
    std_devs_event = std_devs[
        [i for i in range(len(test_dataset.y_upper_bound)) if i not in censored_indices]
    ]
    std_devs_censored = std_devs[censored_indices]
    ax.errorbar(
        true_times_event,
        predicted_times_event,
        yerr=std_devs_event,
        fmt="X",
        capsize=5,
        ecolor=sns_orange,
        markeredgecolor=sns_orange,
        markerfacecolor=sns_orange,
        label="Event",
    )
    ax.errorbar(
        true_times_censored,
        predicted_times_censored,
        yerr=std_devs_censored,
        fmt="X",
        capsize=5,
        ecolor=sns_green,
        markeredgecolor=sns_green,
        markerfacecolor=sns_green,
        label="Censored",
    )
    ax.plot(
        [0, max(true_times_event)],
        [0, max(true_times_event)],
        "--",
        label="Perfect prediction",
        color="black",
    )

    ax.set_xlabel("True survival time (years)")
    ax.set_ylabel("Predicted survival time (years)")
    ax.set_title("True vs Predicted Survival Times in Years")
    ax.legend()

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim(0, max(predicted_times_event))
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis="y", which="both", left=True, right=False, labelbottom=True)
    sns.despine()
    sns.set_style("white")
    plt.tight_layout()
    if not plot_path:
        plt.show()
    else:
        plt.savefig(os.path.join(plot_path, "survival_vs_sd.pdf"))

    ax.cla()
    plt.cla()
    plt.clf()
    plt.close()


def confidence_interval_accuracy(
    y_intervals: tuple[np.ndarray, np.ndarray], y_true: pd.Series | np.ndarray
) -> float:
    """
    Obtain the accuracy of the confidence interval.

    Taken from https://github.com/mvaldenegro/keras-uncertainty.

    :param y_intervals: intervals of the predicted values.
    :param y_true: true values.
    :return: accuracy of the confidence interval.
    """
    interval_min, interval_max = y_intervals
    indicator = np.logical_and(y_true >= interval_min, y_true <= interval_max)
    return np.mean(indicator)


def regressor_calibration_curve(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_std: pd.Series | np.ndarray,
    num_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the reliability plot for a regression prediction.

    Taken from https://github.com/mvaldenegro/keras-uncertainty, with slight modifications.

    :param y_pred: model predictions, usually the mean of the predicted distribution.
    :param y_std: model predicted standard deviation of the predicted distribution.
    :param y_true: ground truth labels.
    :param num_points: number of points in the calibration curve.
    :return: curve_conf, curve_acc: confidence and accuracy values for the calibration curve.
    """
    EPSILON = 1e-5
    alphas = np.linspace(0.0 + EPSILON, 1.0 - EPSILON, num_points + 1)
    curve_conf = []
    curve_acc = []

    for alpha in alphas:
        alpha_intervals = norm.interval(alpha, y_pred, y_std)
        acc = confidence_interval_accuracy(alpha_intervals, y_true)

        curve_conf.append(alpha)
        curve_acc.append(acc)

    return np.array(curve_conf), np.array(curve_acc)


def plot_survival_calibration(
    dataset_test: SurvivalDataset,
    ys: np.ndarray,
    plot_path: str | None = None,
) -> None:
    """
    Plot the calibration curve for survival predictions.

    :param dataset_test: test dataset.
    :param ys: predicted survival times.
    :param plot_path: path to save to, defaults to None
    """
    censored_indices = [
        i
        for i in range(len(dataset_test.y_upper_bound))
        if np.isinf(dataset_test.y_upper_bound[i])
    ]
    true_times_event = dataset_test.y_lower_bound[
        [i for i in range(len(dataset_test.y_upper_bound)) if i not in censored_indices]
    ]
    predicted_times_event = ys[
        :,
        [
            i
            for i in range(len(dataset_test.y_upper_bound))
            if i not in censored_indices
        ],
    ]
    std_devs = predicted_times_event.std(axis=0)
    ypred = predicted_times_event.mean(axis=0)

    conf_curve, acc_curve = regressor_calibration_curve(
        ypred, true_times_event, std_devs
    )
    plt.plot(conf_curve, acc_curve, marker="o")
    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Perfect prediction",
        color="black",
    )
    plt.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    plt.tick_params(axis="y", which="both", left=True, right=False, labelbottom=True)
    plt.xlim(0, 1.1)
    plt.ylim(top=1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    sns.despine()
    sns.set_style("white")
    if not plot_path:
        plt.show()
    else:
        plt.savefig(os.path.join(plot_path, "calibration_curve.pdf"))

    plt.cla()
    plt.clf()
    plt.close()
