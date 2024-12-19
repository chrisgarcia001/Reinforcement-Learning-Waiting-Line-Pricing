# This script allows the plots of an existing study to be redrawn without having to re-run the study script.
# Assumes a study has been pickled and stored somewhere.
# Command Line Usage:
# > python build_hyperparameter_study_plots <path to pickled optuna study file>

import sys
import os
import pickle as pkl

import optuna
from absl import flags
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_intermediate_values


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 2:
        print('Usage: > python build_hyperparameter_study_plots <path to pickled optuna study file>')
        exit()
    study_path = sys.argv[1]
    study = pkl.load(open(study_path, "rb"))
    
    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)
        fig4 = plot_intermediate_values(study)

        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)
    
