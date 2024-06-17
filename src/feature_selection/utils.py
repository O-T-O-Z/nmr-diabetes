import matplotlib.pyplot as plt
import numpy as np


def reduce(feature_selectors: dict) -> dict:
    """
    Select the feature subsets that are equal to the mode of that specific algorithm.

    :param feature_selectors: feature selection results
    :return: feature selection results with the feature subsets that are equal to the mode
    """
    for selector, results in feature_selectors.items():
        lengths = [len(lst) for lst in results["cindex_per_fold"]]
        mode = max(set(lengths), key=lengths.count)
        for inner_key, lst_of_lsts in results.items():
            results[inner_key] = [lst[:mode] for lst in lst_of_lsts]
            results[inner_key] = [lst for lst in lst_of_lsts if len(lst) == mode]
        feature_selectors[selector]["mean_cindex"] = list(
            np.mean(results["cindex_per_fold"], axis=0)
        )
    return feature_selectors


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
