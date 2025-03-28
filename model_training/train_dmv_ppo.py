# This is an adaptation of tune_restaurant_ppo.py. Uses optuna for 5 trials on the optimized model with more learning iterations. 
# Done this way to easily build multiple learning curve plots on the optimized model to characterize variance.


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
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_intermediate_values
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

FLAGS = flags.FLAGS
FLAGS(sys.argv)


study_path = "../models/train_dmv_ppo"
max_request_size = 1
environment_generator_f = ev.dmv_sim_env
rl_control_board_generator_f = sf.RLSinglePriceBoxActionTranslator
min_price = 2
AGENT_ACTION_INTERVAL = 15
STEP_REPORT_INTERVAL = 500
N_CPU = 2
N_TRIALS = 5
LEARN_STEPS = 10000
SEED = 2175

#  Ensure reproducibility of the tuned models.
rnd.seed(SEED)
np.random.seed(2 * SEED)

# Build the sample params for both the RL agent and the control board - came from tuning study.
def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    rl_params =  {'learning_rate': 0.05500934020666769,
                  'gamma': 0.8757263113452977,
                  'ent_coef': 0.0005786611857921733,
                  'n_epochs': 18,
                  'n_steps': 1445}
    cb_params = {'max_reserve_capacity_fraction': 0.5,
                 'max_price': 11.5}
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