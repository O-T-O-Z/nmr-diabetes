# Performs a final cross-validation on the best features found by the feature selection
# algorithm. The best performing feature subset is saved to a file.

import json
import os

from tqdm import tqdm

from runner_functions import cross_validate_model
from survival_dataset import SurvivalDataset
from utils import load_feature_selection

if __name__ == "__main__":
    SAVE_PATH = "../../results_fs"
    with open(os.path.join(SAVE_PATH, "best_params_xgb_aft.json"), "r") as f:
        best_params_aft = json.load(f)

    total_set = []
    for ds_name in ["clinical", "nmr"]:
        dataset = SurvivalDataset()
        dataset.load_data(f"{ds_name}_train.csv")
        subsets = load_feature_selection(
            os.path.join(SAVE_PATH, f"{ds_name}_featureselection.json"), dataset.X
        )

        res = []
        for subset in tqdm(subsets):
            dataset_copy = dataset.model_copy()
            dataset_copy.X = dataset_copy.X[subset]
            result = cross_validate_model(
                dataset,
                best_params_aft,
                k_fold_seed=0,
                early_stopping_rounds=50,
                n_folds=10,
            )
            res.append((subset, result))

        best_subset, best_result = max(res, key=lambda x: x[1])
        print(best_subset, best_result)
        with open(os.path.join(SAVE_PATH, f"{ds_name}_best_features.txt"), "w") as f:
            f.write(f"{best_subset}")
        total_set.extend(best_subset)
    with open(os.path.join(SAVE_PATH, "full_best_features.txt"), "w") as f:
        f.write(f"{total_set}")
