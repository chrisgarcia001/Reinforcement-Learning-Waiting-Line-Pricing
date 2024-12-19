import sys
sys.path.insert(1, '../')

import simpy
import numpy as np
import sim_framework as sf
import random
import pprint

#RANDOM_SEED = 216
#random.seed(RANDOM_SEED)

SIM_DAY_LENGTH = 60 * 14 # 10am to midnight = 14 hours 
AM_11_30 = 90
PM_1_00 = 180
PM_6_00 = 60 * 8
PM_9_00 = 60 * 11
AM_12_00 = SIM_DAY_LENGTH

TOTAL_CAPACITY = 50

simpy_env = simpy.Environment()

sim_logger = sf.BasicSimLog(simpy_env)#, 30)

balk_threshold_dist = sf.NormalDist(60, 15, lower_clip=15, upper_clip=180).generate
premium_threshold_lower_dist = sf.NormalDist(5, 2, lower_clip=3, upper_clip=12).generate
premium_upper_lower_delta_dist = sf.NormalDist(40, 8, lower_clip=10, upper_clip=50).generate
max_premium_unit_price_dist = sf.NormalDist(15, 8, lower_clip=2).generate
min_acceptable_wait_reduction_percent_dist = sf.UniformDist(0.6, 0.9, False).generate

customer_decision_model_generator = sf.CustomerDecisionModelGenerator(balk_threshold_dist,           
                                                premium_threshold_lower_dist,                   
                                                premium_upper_lower_delta_dist,                
                                                max_premium_unit_price_dist,                    
                                                min_acceptable_wait_reduction_percent_dist)
customer_decision_model_f = customer_decision_model_generator.generate_decision_f()

max_request_size = 10
price_capacity_rules = [(AM_11_30, (0, 0)), (PM_1_00, (5, 6)), (PM_6_00, (0, 0)),
                        (PM_9_00, (7, 10)), (AM_12_00, (0, 0))]
                        
control_board = sf.SimTimeRuleControlBoard(simpy_env, max_request_size, price_capacity_rules)

base_service_time_f = lambda x: x * sf.NormalDist(8, 2, lower_clip=3, upper_clip=15, round_digits=0).generate()
base_revenue_f = lambda x: x * sf.NormalDist(8, 2, lower_clip=3, upper_clip=15, round_digits=1).generate()
mean_arrival_time_phases = [(AM_11_30, sf.ExpInterArrivalTimeDist(5)), 
                            (PM_1_00, sf.ExpInterArrivalTimeDist(15)), 
                            (PM_6_00, sf.ExpInterArrivalTimeDist(5)),
                            (PM_9_00, sf.ExpInterArrivalTimeDist(40)), 
                            (AM_12_00, sf.ConstantDist(180))]
interarrival_time_f = sf.SimTimeDependentDist(simpy_env, mean_arrival_time_phases).generate
#interarrival_time_f = sf.ConstantDist(2).generate
request_size_f = sf.HistogramDist({1:5, 2:20, 3:10, 4:15, 5:5, 6:5, 7:2, 8:1}).generate
loiter_time_f = sf.NormalDist(20, 5, lower_clip=8, round_digits=0).generate
sim_day_length = SIM_DAY_LENGTH
agent_action_interval = 15



TOTAL_CONTROL_ITERATIONS = 25 * 14 * 4 # (1 year = 365 days) * (14 hours open per day) * (4 15-minute intervals per hour)
wait_time_pricing_env = sf.WaitTimePricingSimEnv(TOTAL_CAPACITY, 
                 base_service_time_f, 
                 base_revenue_f,
                 interarrival_time_f,
                 request_size_f,
                 loiter_time_f,
                 customer_decision_model_generator,
                 agent_action_interval,
                 sim_day_length,
                 sim_logger,
                 simpy_env,
                 control_board,
                 step_report_interval=15)


state = wait_time_pricing_env.reset()
i = 0
while i < TOTAL_CONTROL_ITERATIONS:
    state, reward, done, info = wait_time_pricing_env.step(np.float32([0]))
    if done:
        print('RESETTING...')
        wait_time_pricing_env.reset()
    i += 1


balks = [x for x in sim_logger.get_history() if 'event' in x and x['event'] == 'customer_decision' and x['decision'] == 'balk']
print('Balks:', len(balks))
regular_takes = [x for x in sim_logger.get_history() if 'event' in x and x['event'] == 'customer_decision' and x['decision'] == 'regular']
print('Regular Takes:', len(regular_takes))
premium_takes = [x for x in sim_logger.get_history() if 'event' in x and x['event'] == 'customer_decision' and x['decision'] == 'premium']
print('Premium Takes:', len(premium_takes))

c_balks = sum([x[2] for (x, _) in wait_time_pricing_env.state_reward_history])
c_regular_takes = sum([x[4] for (x, _) in wait_time_pricing_env.state_reward_history])
c_premium_takes = sum([x[3] for (x, _) in wait_time_pricing_env.state_reward_history])
assert(len(balks) == c_balks)
assert(len(regular_takes) == c_regular_takes)
assert(len(premium_takes) == c_premium_takes)
print('------ ALL DONE: SUCCESS! -----------')



