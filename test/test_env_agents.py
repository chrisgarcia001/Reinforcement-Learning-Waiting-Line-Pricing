import sys
sys.path.insert(1, '../')

import simpy
import numpy as np
import sim_framework as sf
import environments as ev
import rule_based_agents as ag
from stable_baselines3.common.env_checker import check_env

simpy_env = simpy.Environment()

AGENT_ACTION_INTERVAL = 15

#agent, control_board = ag.build_restaurant_no_premium_agent(simpy_env)

#env, sim_logger, hours_per_day = ev.restaurant_sim_env(simpy_env, agent_action_interval=AGENT_ACTION_INTERVAL)
#agent, control_board = ag.build_restaurant_rule_agent(env) 

#env, sim_logger, hours_per_day = ev.dmv_sim_env(simpy_env, agent_action_interval=AGENT_ACTION_INTERVAL) 
#agent, control_board = ag.build_dmv_rule_agent(env) 

env, sim_logger, hours_per_day = ev.amusement_park_ride_sim_env(simpy_env, agent_action_interval=AGENT_ACTION_INTERVAL) 
agent, control_board = ag.build_amusement_park_rule_agent(env)

check_env(env)
#exit()


TOTAL_CONTROL_ITERATIONS = int(25 * hours_per_day * (60 / AGENT_ACTION_INTERVAL)) # (days in horizon) * (hours operating per day) * (number of action/update intervals per hour)

state = env.reset()

for i in range(TOTAL_CONTROL_ITERATIONS):
    action, _state = agent.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    if done:
        print('RESETTING...')
        state = env.reset()


balks = [x for x in sim_logger.get_history() if 'event' in x and x['event'] == 'customer_decision' and x['decision'] == 'balk']
print('Balks:', len(balks))
regular_takes = [x for x in sim_logger.get_history() if 'event' in x and x['event'] == 'customer_decision' and x['decision'] == 'regular']
print('Regular Takes:', len(regular_takes))
premium_takes = [x for x in sim_logger.get_history() if 'event' in x and x['event'] == 'customer_decision' and x['decision'] == 'premium']
print('Premium Takes:', len(premium_takes))
base_rev = sum([x['base_revenue'] for x in sim_logger.get_history() if 'event' in x and x['event'] and x['event'] == 'service_complete'])
prem_rev = sum([x['premium_revenue'] for x in sim_logger.get_history() if 'event' in x and x['event'] and x['event'] == 'service_complete'])
print('Base, Premium Revenue:', base_rev, prem_rev)