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
