# This script runs an experiment using the restaurant scenario to compare the the following policies:
# 1) A control policy, using no premium pricing
# 2) A simple rule-based policy that alters price based on the time of day
# 3) A trained RL agent policy

import sys
sys.path.insert(1, '../')
import random as rnd
import numpy as np
import time

import simpy
import sim_framework as sf
import environments as ev
import rule_based_agents as ag

from stable_baselines3 import PPO

# Key experiment parameters
results_csv_path = "../results/amusement_park_results.csv"
trained_model_path = "../models/train_amusement_park_ppo/trial_2/best_model"
params_path = "../models/train_amusement_park_ppo/trial_2/params.txt"
max_request_size = 1
environment_generator_f = ev.amusement_park_ride_sim_env
rl_single_price_control_board_generator_f = sf.RLSinglePriceBoxActionTranslator
baseline_agent_builder_f = ag.build_amusement_park_no_premium_agent
rule_agent_builder_f = ag.build_amusement_park_rule_agent
min_price = 2
AGENT_ACTION_INTERVAL = 15
STEP_REPORT_INTERVAL = 500
SEED = 6218 #  To ensure reproducibility and identical conditions in each trial.
DAYS_PER_TRIAL = 30


# Read in params to get control board params
params = {}
with open(params_path) as f:
    param_text = f.read()
    params = eval(param_text)

max_reserve_capacity_fraction = params['cb_params']['max_reserve_capacity_fraction']
max_price = params['cb_params']['max_price']

# Constructs identical/parallel environments for each agent being tested.
def construct_trials():
    s1 = simpy.Environment()
    env_1, sim_log_1, hours_per_day = environment_generator_f(s1, agent_action_interval=AGENT_ACTION_INTERVAL, step_report_interval=STEP_REPORT_INTERVAL)
    a1, cb1 = baseline_agent_builder_f(env_1)
    
    s2 = simpy.Environment()
    env_2, sim_log_2, hours_per_day = environment_generator_f(s2, agent_action_interval=AGENT_ACTION_INTERVAL, step_report_interval=STEP_REPORT_INTERVAL)
    a2, cb2 = rule_agent_builder_f(env_2)
    
    s3 = simpy.Environment()
    env_3, sim_log_3, hours_per_day = environment_generator_f(s3, agent_action_interval=AGENT_ACTION_INTERVAL, step_report_interval=STEP_REPORT_INTERVAL)
    if env_3.two_class_service == None:
        env_3.reset()
    tc = env_3.two_class_service
    cb3 = rl_single_price_control_board_generator_f(tc.capacity, 0, int(round(tc.capacity * max_reserve_capacity_fraction, 0)),
                                                    max_request_size, max_price, min_price=min_price) 
    env_3.set_control_board(cb3)
    a3 = PPO.load(trained_model_path, env=env_3)
    t1 = {'name':'NO_PREMIUM_POLICY', 'model':a1, 'control_board':cb1, 'sim_log':sim_log_1, 'env':env_1, 'seed':SEED, 'hours_per_day':hours_per_day}
    t2 = {'name':'RULE_BASED_POLICY', 'model':a2, 'control_board':cb2, 'sim_log':sim_log_2, 'env':env_2, 'seed':SEED, 'hours_per_day':hours_per_day}
    t3 = {'name':'PPO_POLICY', 'model':a3, 'control_board':cb3, 'sim_log':sim_log_3, 'env':env_3, 'seed':SEED, 'hours_per_day':hours_per_day}
    trials = [t1, t2, t3]
    return trials
 
 
# Compute key statistics after a trial to be used for comparision.
def compile_statistics(sim_logger):
    balks = len([x for x in sim_logger.get_history() if 'event' in x and x['event'] == 'customer_decision' and x['decision'] == 'balk'])
    regular_takes = len([x for x in sim_logger.get_history() if 'event' in x and x['event'] == 'customer_decision' and x['decision'] == 'regular'])
    premium_takes = len([x for x in sim_logger.get_history() if 'event' in x and x['event'] == 'customer_decision' and x['decision'] == 'premium'])
    base_rev = sum([x['base_revenue'] for x in sim_logger.get_history() if 'event' in x and x['event'] == 'service_complete'])
    prem_rev = sum([x['premium_revenue'] for x in sim_logger.get_history() if 'event' in x and x['event'] == 'service_complete'])
    total_rev = sum([x['total_revenue'] for x in sim_logger.get_history() if 'event' in x and x['event'] == 'service_complete'])
    return {'balks':balks, 'regular_takes':regular_takes, 'premium_takes':premium_takes,
            'base_revenue':base_rev, 'premium_revenue':prem_rev, 'total_revenue':total_rev}


# Run the experiment, generate statistics, and save them into the specified CSV file.    
def run():
    trials = construct_trials()
    stats = []
    results = ['Policy,Balks,Regular Takes,Premium Takes,Base Revenue,Premium Revenue,Total Revenue']
    for t in trials:
        rnd.seed(SEED)           # Ensure exact reproducibility/same scenario for each model/policy
        np.random.seed(2 * SEED) # Also needs to be done for underlying distributions used in scipy
        model = t['model']
        env = t['env']
        obs = env.reset()
        trial_iterations = DAYS_PER_TRIAL * t['hours_per_day'] * AGENT_ACTION_INTERVAL
        print('++++++++++++ STARTING TRIAL:', t['name'], '++++++++++++')
        for i in range(trial_iterations):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        stat = compile_statistics(t['sim_log'])
        print('CONCLUDING TRIAL:', t['name'])
        print('  TRIAL RESULTS:')
        print('      Balks:', stat['balks'])
        print('      Regular Takes:', stat['regular_takes'])
        print('      Premium Takes:', stat['premium_takes'])
        print('      Base Revenue:', stat['base_revenue'])
        print('      Premium Revenue:', stat['premium_revenue'])
        print('      Total Revenue:', stat['total_revenue'])
        s = ','.join([str(t['name']), str(stat['balks']), str(stat['regular_takes']), str(stat['premium_takes']), 
                       str(stat['base_revenue']), str(stat['premium_revenue']), str(stat['total_revenue'])])
        results.append(s)
    with open(results_csv_path, 'w') as outfile:
        outfile.write("\n".join(results))
    print("\n  -- Experimental Trials Complete! Results Written To:", results_csv_path, '--')


if __name__ == "__main__":
    run()