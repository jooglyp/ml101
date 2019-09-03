# Binary classification machine learning model(s) for loan default events
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
.pyenv/bin/tox
```

## Generate the latest dependencies (to update requirements.txt)
```bash
# Using this repository as the current working directory
#a
.pyenv/bin/pip-compile -vvv --upgrade --dry-run setup.py
```