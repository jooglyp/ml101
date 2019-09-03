# Binary classification machine learning model(s) for loan default events

## This repository was written and tested on a Debian Ubuntu Linux machine running PopOS

## Setup (Editable)
```bash
# Using this repository as the current working directory
python3.7 -m venv .pyenv  # Install Python 3.7 virtual environment
.pyenv/bin/pip install --upgrade pip wheel pip-tools bumpversion tox  # Install additional tools
.pyenv/bin/pip install -e .  # Install this application (in editable mode)
```

## Run the unit tests
```bash
# Using this repository as the current working directory
# Note: the dask scheduler does not print to stdout because it's running in a different process.
.pyenv/bin/tox
```

## Generate the latest dependencies (to update requirements.txt)
```bash
# Using this repository as the current working directory
.pyenv/bin/pip-compile -vvv --upgrade --dry-run setup.py
```

## Recommendation grid search parameter for model.py XGBoostModel.tune_parameters():
```
# Recommendation for model demonstration:
"""
param_grid = [
    ("grid_neighbors", [2]),
    ("grid_sample_proportion", [0.9]),
    ("category_limit", [10]),
    ("pca_proportion", [0.95]),
    ("pca_components", [4])
]
"""
```

## To run the model locally:
```
# Copy the assignment data to /tmp/
# cd to project directory
.pyenv/bin/ml101
```

## Notes on client-side integration:
```buildoutcfg
# Please feed an X test set that resembles the training set.
# Since data transformation is only applied during sampling, I will go ahead 
# and apply the sampler to the client's data and treat it as if it were being split.
# But it will mean that the dimensions of the data will be chosen by the algorithm.
```

## Notes on dask implementation:
```buildoutcfg
# dask distributed pipes break when run-time is complete and cause warnings in tests.
# These bugs are due to dask having difficulty managing local threads.
```

## Notes on general model framework:
```buildoutcfg
# class 1: PCA to identify best variables
# class 2: k-fold cross-validator/imbalanced sampling
# class 3: Dask XGBoost
# class 4: generate accuracy scores
# - f1 (precision versus recall) - and confusion matrix
# - log-loss
# class 5: optimization using pca and sampling in a custom grid search framework
```