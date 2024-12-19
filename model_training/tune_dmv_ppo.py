# This script uses optuna to find approximately optimal hyperparameter settings for the restaurant PPO algorithm. 
# This was adapted from the following example:
#        https://github.com/CppMaster/SC2-AI/blob/master/minigames/move_to_beacon/src/optuna_search.py

import sys
sys.path.insert(1, '../')

import simpy
import sim_framework as sf
import environments as ev

from typing import Optional

import os
import pickle as pkl
import random
import sys
import time
from pprint import pprint
from typing import Any, Dict
import random as rnd
import numpy as np

import gym

import optuna
from absl import flags
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

FLAGS = flags.FLAGS
FLAGS(sys.argv)


study_path = "../models/tune_dmv_ppo"
max_request_size = 1
environment_generator_f = ev.dmv_sim_env
rl_control_board_generator_f = sf.RLSinglePriceBoxActionTranslator
min_price = 2
AGENT_ACTION_INTERVAL = 15
STEP_REPORT_INTERVAL = 500
N_CPU = 2
N_TRIALS = 100
LEARN_STEPS = 5000
SEED = 3838

#  Ensure reproducibility of the tuned models.
rnd.seed(SEED)
np.random.seed(2 * SEED)

# Build the sample params for both the RL agent and the control board.
def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1, log=True)
    gamma = trial.suggest_float("gamma", 0.85, 0.99999, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    n_epochs = trial.suggest_int('n_epochs', 1, 25)
    n_steps = trial.suggest_int('n_steps', 100, 2500)
    max_reserve_capacity_fraction = trial.suggest_discrete_uniform("max_reserve_capacity_fraction", 0.1, 0.5, 0.05)
    max_price = trial.suggest_float("max_price", 5, 40, step=0.5, log=False)
    rl_params =  {'learning_rate': learning_rate, 
                  'gamma': gamma, 
                  'ent_coef': ent_coef, 
                  'n_epochs': n_epochs,
                  'n_steps': n_steps}
    cb_params = {'max_reserve_capacity_fraction': max_reserve_capacity_fraction,
                 'max_price': max_price}
    return {'rl_params': rl_params, 'cb_params':cb_params}


# -------------------------------------------------------------------------
# This callback component was taken adapted from the following:
#      https://github.com/CppMaster/SC2-AI/blob/master/optuna_utils/trial_eval_callback.py
class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
            best_model_save_path: Optional[str] = None,
            log_path: Optional[str] = None,
            callback_after_eval: Optional[BaseCallback] = None
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            callback_after_eval=callback_after_eval
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            continue_training = super()._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elapsed time ?
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return continue_training
# -------------------------------------------------------------------------



def objective(trial: optuna.Trial) -> float:
    env_kwargs = {} #{"step_mul": step_mul}

    sampled_hyperparams = sample_params(trial) 
    rl_params, cb_params = sampled_hyperparams['rl_params'], sampled_hyperparams['cb_params']
    
    max_reserve_capacity_fraction = cb_params['max_reserve_capacity_fraction']
    max_price = cb_params['max_price']

    path = f"{study_path}/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)
    
    simpy_env = simpy.Environment()
    env, sim_log, hours_per_day = environment_generator_f(simpy_env, agent_action_interval=AGENT_ACTION_INTERVAL, step_report_interval=STEP_REPORT_INTERVAL)
    if env.two_class_service == None:
        env.reset()
    tc = env.two_class_service
    control_board = rl_control_board_generator_f (tc.capacity, 0, int(round(tc.capacity * max_reserve_capacity_fraction, 0)),
                                                  max_request_size, max_price, min_price=min_price) 
    env.set_control_board(control_board)
    env = Monitor(env)
    model = PPO("MlpPolicy", env=env, seed=None, verbose=0, **rl_params)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=50, verbose=1)
    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=5, eval_freq=500, deterministic=False, callback_after_eval=stop_callback
    )

    params = sampled_hyperparams #env_kwargs | sampled_hyperparams
    with open(f"{path}/params.txt", "w") as f:
        f.write(str(params))

    try:
        model.learn(LEARN_STEPS, callback=eval_callback)
        env.close()
    except (AssertionError, ValueError) as e:
        env.close()
        print(e)
        print("============")
        print("Sampled params:")
        pprint(params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.best_mean_reward

    del model.env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


if __name__ == "__main__":

    sampler = TPESampler(n_startup_trials=10, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    try:
        study.optimize(objective, n_jobs=N_CPU, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{study_path}/report.csv")

    with open(f"{study_path}/study.pkl", "wb+") as f:
        pkl.dump(study, f)

    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)

        fig1.show()
        fig2.show()
        fig3.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)