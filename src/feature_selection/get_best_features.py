# Performs a final cross-validation on the best features found by the feature selection
# algorithm. The best performing feature subset is saved to a file.

import json
import os

from runner_functions import cross_validate_model
from src.utils import load_feature_selection
from survival_dataset import SurvivalDataset

if __name__ == "__main__":
    SAVE_PATH = "../../results_fs"
    with open(os.path.join(SAVE_PATH, "best_params_xgb_aft.json"), "r") as f:
        best_params_aft = json.load(f)

    for ds_name in ["clinical", "nmr"]:
        dataset = SurvivalDataset()
        dataset.load_data(f"{ds_name}_train.csv")
        subsets = load_feature_selection(
            os.path.join(SAVE_PATH, f"{ds_name}_featureselection.json"), dataset.X
        )

        res = []
        for subset in subsets:
            result = cross_validate_model(
                dataset.X[subset],
                dataset.y_lower_bound,
                dataset.y_upper_bound,
                best_params_aft,
                dataset.censored,
                k_fold_seed=0,
                early_stopping_rounds=50,
                n_folds=10,
            )
            res.append((subset, result))

        best_subset, best_result = max(res, key=lambda x: x[1])
        print(best_subset, best_result)
        with open(os.path.join(SAVE_PATH, "{ds_name}_best_features.txt"), "w") as f:
            f.write(f"{best_subset}")
