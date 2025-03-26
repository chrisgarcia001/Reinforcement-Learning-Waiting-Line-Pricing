# Reinforcement-Learning-Waiting-Line-Pricing
Research Code for RL-Based Dynamic Pricing and Capacity Allocation in Monetized Wait-Skipping Services

This repository contains the core source code, experiments, and results from my research paper entitled "Reinforcement learning for dynamic pricing and capacity allocation in monetized customer wait-skipping services". The code is written in Python 3. In order to run any code or excute the experiments, make sure to first pip install the packages listed in "requirements.txt". This repository is organized as follows:

1. Core Source Code

The core source code is found in the following files:

- **sim_framework.py:** This file contains the main simulation abstractions and component implementations (implemented using the *simpy* package) used for training the RL models and executing the experiments.
- **rule_based_agent.py:** This file contains the human-engineered policy implementations used for comparison in the experiments.
- **environments.py:** This file contains the environment configurations for the three test scenarios (Theme Park, DMV, and Restaurant) used in the research paper.
- **util.py:** This file contains utility components used in training, performance evaluation, and analysis.
- **build_hyperparameter_study_plots.py**: This script allows the plots of an existing Optuna study to be drawn without having to re-run the study script.

2. Model Tuning and Training

The code for hyperparameter tuning and training the models used in each scenario is found in the *model_training* folder. When each of these files are run,
a correspondingly-named Stable Baselines 3 PPO model is trained and placed within the the *models* folder. 

3.  Computational Experiments

The code for running each of the three simulation experiments can be found in the *experiments* folder. Note that the models must be trained before this code can be executed.

4.  Results

The experimental results generated for each experiment can be found under the *results* folder.

### Paper and Citation

If you wish to cite this work, please use the following reference:

Garcia, C. (2025), ["Reinforcement learning for dynamic pricing and capacity allocation in monetized customer wait-skipping services"](https://doi.org/10.1080/2573234X.2024.2424542), *Journal of Business Analytics*, Vol. 8, No. 1, pp. 36-54.
