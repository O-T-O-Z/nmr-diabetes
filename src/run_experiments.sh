#!/bin/bash
python run_experiments.py --load-features best_features --save experiment_normal
python run_experiments.py --load-features best_features --save experiment_normal_mcc --use-mcc
python run_experiments.py --load-features best_features --save experiment_normal_bagging --bagging
python run_experiments.py --load-features best_features --save experiment_normal_mcc_bagging --use-mcc --bagging
