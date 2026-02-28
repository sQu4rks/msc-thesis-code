To redo an experiment you have to install some dependencies - there is a pyproject.toml for a uv project in the code directory
that installs the required dependencies. Navigate to the code directory and run uv sync to setup a new environment with the 
dependencies. 

This has been tested only under Mac OS X 15.7.3 with uv 

1. Run the generate_data.py file. The easiest way to run it is using uv and then

> uv run generate_data.py --help 

for all available options.

- use the --federated flag to save per-node data for federated training and the --split flag to split into train/val/test
- for the experiments, a --total-traffic 3000 was used to introduce bottlenecks
- otherwise the default values should represent those used during the experiments

2. Train the centralized model. 

> uv run train_centralized.py --data-dir <path to data dir from previous generation command>

3. Evaluate the model 

> uv run train_centralized.py --evaluate 

- this will use the best saved model 

4. Train the federated models

For each of the federated models it is easiest to use the flower cli to run them. Each directory prefixed with "federated-"
contains one of the four experiments. 

Navigate into the folder (i.e. federated-oracle) and create a venv for this specific experiment with 

> uv venv --python 3.13 

and install the dependencies

> uv pip install "flwr[simulation]" torch numpy

and run the training of the model with 

> source .venv/bin/activate && flwr run . local-simulation

Note: The pinning to python version 3.13 worked to get around an import error with ray (see https://github.com/adap/flower/issues/5512).

5. Use the evaluation script to evaluate the different models

Navigate to the core directory and run the evaluation script with

> uv run evaluate_offline.py --data-dir ../data/training_data --centralized-model <path to checkpoint> --federated-dir ../federated-oracle --federated-local-dir ../federated-local --federated-neighbor-dir ../federated-neighbor --federated-linkutil-dir ../federated-linkutil

- ECMP and Oracle baselines are always computed
- The federated models are optional. Toggle them by including their respective data dir argument (i.e. --federated-local-dir for the federated-local model)
