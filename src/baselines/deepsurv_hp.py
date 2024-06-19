# Original DeepSurv hyperparameter optimization code
import argparse
import json
import logging
import os
import pickle
import sys
import uuid

import lasagne
import numpy as np
import optunity
from deepsurv import DeepSurv
from numpy import float32, int32

from src.runner_functions import SurvivalDataset
from utils import load_best_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="Directory for tensorboard logs")
    parser.add_argument("dataset", help="Dataset to load")
    parser.add_argument(
        "box", help="Filename to box constraints dictionary pickle file"
    )
    parser.add_argument("num_evals", help="Number of models to test", type=int)
    parser.add_argument("--update_fn", help="Lasagne optimizer", default="sgd")
    parser.add_argument(
        "--num_epochs", type=int, help="Number of epochs to train", default=100
    )
    parser.add_argument(
        "--num_folds", type=int, help="Number of folds to cross-validate", default=5
    )
    return parser.parse_args()


def load_logger(logdir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Print to Stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    logger.addHandler(ch)

    # Print to Log file
    fh = logging.FileHandler(os.path.join(logdir, "log_" + str(uuid.uuid4())))
    fh.setFormatter(format)
    logger.addHandler(fh)

    return logger


def format_to_optunity(dataset, strata=False):
    """
    Formats a dataset dictionary containing survival data with keys:
        {
            'x' : baseline data
            'e' : censor
            't' : event time
        }
    to a format that Optunity can use to run hyper-parameter searches on.
    """
    x = dataset["x"]
    e = dataset["e"]
    t = dataset["t"]
    y = np.column_stack((e, t))
    # Take the indices of censored entries as strata
    if strata:
        strata = [np.nonzero(np.logical_not(e).astype(np.int32))[0].tolist()]
    else:
        strata = None
    return (x, y, strata)


def load_box_constraints(file):
    with open(file, "rb") as fp:
        return json.loads(fp.read())


def save_call_log(file, call_log):
    with open(file, "wb") as fp:
        pickle.dump(call_log, fp)


def get_objective_function(num_epochs, logdir, update_fn=lasagne.updates.sgd):
    """
    Returns the function for Optunity to optimize. The function returned by get_objective_function
    takes the parameters: x_train, y_train, x_test, and y_test, and any additional kwargs to
    use as hyper-parameters.

    The objective function runs a DeepSurv model on the training data and evaluates it against the
    test set for validation. The result of the function call is the validation concordance index
    (which Optunity tries to optimize)
    """

    def format_to_deepsurv(x, y):
        return {"x": x, "e": y[:, 0].astype(np.int32), "t": y[:, 1].astype(np.float32)}

    def get_hyperparams(params):
        hyperparams = {
            "batch_norm": True,
            "activation": "rectify",
            "standardize": False,
        }
        # @TODO add default parameters and only take necessary args from params
        # protect from params including some other key

        if "num_layers" in params and "num_nodes" in params:
            params["hidden_layers_sizes"] = [int(params["num_nodes"])] * int(
                params["num_layers"]
            )
            del params["num_layers"]
            del params["num_nodes"]

        if "learning_rate" in params:
            params["learning_rate"] = 10 ** params["learning_rate"]

        hyperparams.update(params)
        return hyperparams

    def train_deepsurv(x_train, y_train, x_test, y_test, **kwargs):
        # Standardize the datasets
        # train_mean = x_train.mean(axis=0)
        # train_std = x_train.std(axis=0)

        # x_train = (x_train - train_mean) / train_std
        # x_test = (x_test - train_mean) / train_std

        train_data = format_to_deepsurv(x_train, y_train)
        valid_data = format_to_deepsurv(x_test, y_test)

        hyperparams = get_hyperparams(kwargs)

        # Set up Tensorboard loggers
        # TODO improve the model_id for Tensorboard to better partition runs
        model_id = str(hash(str(hyperparams)))
        run_id = model_id + "_" + str(uuid.uuid4())

        network = DeepSurv(n_in=x_train.shape[1], **hyperparams)
        metrics = network.train(
            train_data,
            n_epochs=num_epochs,
            # update_fn=update_fn,
            verbose=False,
        )
        result = network.get_concordance_index(**valid_data)

        main_logger.info(
            "Run id: %s | %s | C-Index: %f | Train Loss %f"
            % (run_id, str(hyperparams), result, metrics["loss"][-1])
        )
        return result

    return train_deepsurv


if __name__ == "__main__":
    NUM_EPOCHS = 500
    NUM_FOLDS = 3
    ds_name = "full"
    global main_logger
    main_logger = load_logger("../logdir/")

    dataset = SurvivalDataset()
    dataset.load_data(f"{ds_name}_train.csv")
    if ds_name in ["clinical", "nmr"]:
        best_features = load_best_features(f"../results_fs/{ds_name}_best_features.txt")
    else:
        best_features = load_best_features(
            "../results_fs/clinical_best_features.txt"
        ) + load_best_features("../results_fs/nmr_best_features.txt")
    dataset.X = dataset.X[best_features]

    train_data = {
        "x": dataset.X.values.astype(float32),
        "t": dataset.y_lower_bound.values.astype(float32),
        "e": dataset.event.values.astype(int32),
    }
    x, y, strata = format_to_optunity(train_data)

    box_constraints = load_box_constraints(
        "/Users/otoz/Desktop/DeepSurv/hyperparam_search/box_constraints.0.json"
    )
    main_logger.debug("Box Constraints: " + str(box_constraints))

    opt_fxn = get_objective_function(NUM_EPOCHS, "logdir/", lasagne.updates.adam)
    opt_fxn = optunity.cross_validated(x=x, y=y, num_folds=NUM_FOLDS, strata=strata)(
        opt_fxn
    )

    main_logger.debug("Maximizing C-Index. Num_iterations: %d" % 200)
    from tqdm import tqdm

    opt_params, call_log, _ = tqdm(
        optunity.maximize(opt_fxn, num_evals=50, solver_name="sobol", **box_constraints)
    )

    main_logger.debug("Optimal Parameters: " + str(opt_params))
    main_logger.debug("Saving Call log...")
    with open(f"../results_models/deepsurv/{ds_name}_best_params.json", "w") as fp:
        json.dump(opt_params, fp)
    print(call_log._asdict())
    save_call_log(
        os.path.join("../logdir/", "optunity_log_%s.pkl" % (str(uuid.uuid4()))),
        call_log._asdict(),
    )
