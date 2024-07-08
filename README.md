# NMR Diabetes Prediction
The repository containing the code for the thesis titled: "Machine Learning Based Survival Analysis Integrating Blood NMR Data for Explainable Type 2 Diabetes Prediction".

## Setting up environment

To get started, please install the requirements on a Python 3.10 environment. An example using `conda`:

```shell
conda create -n venv python=3.11
conda activate venv
pip install -r requirements.txt
```

## Structure
The repository is structured as follows:
- `results_fs` contains all results regarding the feature selection experiment (Experiment 1).
- `results_models` contains all results regarding the model experiments (Experiments 2-4).
- `methods` contains all plots used in the thesis methods.
- `src` contains all code used to run all experiments.

## Usage notice
Due to the proprietary nature of the datasets used in the thesis, none were included here. For the sake of reproducability
and transparency, it was decided to still publish the code. If any code or analysis is found to be incorrect, please
feel free to open an issue or contact the author.
