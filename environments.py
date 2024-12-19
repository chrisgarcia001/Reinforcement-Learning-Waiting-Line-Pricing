# This file contains functions to generate different characteristic service environments
# for training and/or testing agents controlling the pricing/capacity allocation for wait skipping options.

import simpy
import numpy as np
import sim_framework as sf


# This function builds a restaurant simulation environment with varying party sizes (i.e. request sizes).
# The restaurant receives customers from 10am to 9 pm with varying customer arrival rates at
# different times throughout the day, with peaks at lunch and dinner times. 
# @param agent_action_interval: the number of simulation time minutes representing the time periond in which
#                               the control agent updates premium prices and capacity reservation.
# @returns: a tuple of form (WaitTimePricingSimEnv, BasicSimLog, Operating Hours per Day) 
def restaurant_sim_env(simpy_env, agent_action_interval=15, step_report_interval=15):
    SIM_DAY_LENGTH = 60 * 14 # 10am to midnight = 14 hours (Note: customers stop coming by 9pm).
    AM_11_30 = 60 * 1.5 
    PM_1_00 = 60 * 3
    PM_6_00 = 60 * 8
    PM_8_00 = 60 * 10
    PM_9_00 = 60 * 11
    AM_12_00 = SIM_DAY_LENGTH

    TOTAL_CAPACITY = 50 # Restaurant has 50 seat capacity

    sim_logger = sf.BasicSimLog(simpy_env)

    # Customers balk after average of 65 minutes with SD of 15
    balk_threshold_dist = sf.NormalDist(65, 15, lower_clip=15, upper_clip=180).generate
        
    # Customers reach the maximum willingness to pay for premium after 45 minute waith time with SD of 12 minutes.
    premium_threshold_lower_dist = sf.ConstantDist(0).generate
    premium_upper_lower_delta_dist = sf.NormalDist(45, 12, lower_clip=10).generate
    
    # Customers are willing to pay a maximum of 15 per premium seat with SD of 8
    max_premium_unit_price_dist = sf.NormalDist(15, 8, lower_clip=2).generate
    
    # Customers expect at least 60-90% reduction in waiting time in order to pay for premium
    min_acceptable_wait_reduction_percent_dist = sf.UniformDist(0.6, 0.9, False).generate

    customer_decision_model_generator = sf.CustomerDecisionModelGenerator(balk_threshold_dist,           
                                                    premium_threshold_lower_dist,                   
                                                    premium_upper_lower_delta_dist,                
                                                    max_premium_unit_price_dist,                    
                                                    min_acceptable_wait_reduction_percent_dist)
    customer_decision_model_f = customer_decision_model_generator.generate_decision_f()

    # Base service (i.e. preparing meals) is 8 minutes per seat with SD of 2
    base_service_time_f = lambda x: x * sf.NormalDist(8, 2, lower_clip=3, upper_clip=15, round_digits=0).generate()
    
    # Base revenue (i.e. meal costs) is 20 dollars with SD of 5
    base_revenue_f = lambda x: x * sf.NormalDist(20, 5, lower_clip=3, round_digits=1).generate()
    
    # The mean arrival rates for different time ranges. Peak at lunch (11:30 am - 1:00 pm) and dinner (6:00 - 8:00 pm).
    mean_arrival_time_phases = [(AM_11_30, sf.ExpInterArrivalTimeDist(5)), 
                                (PM_1_00, sf.ExpInterArrivalTimeDist(12)), 
                                (PM_6_00, sf.ExpInterArrivalTimeDist(5)),
                                (PM_8_00, sf.ExpInterArrivalTimeDist(25)),
                                (PM_9_00, sf.ExpInterArrivalTimeDist(3)), 
                                (AM_12_00, sf.ConstantDist(180))]
    interarrival_time_f = sf.SimTimeDependentDist(simpy_env, mean_arrival_time_phases).generate
    
    # Party sizes and relative frequency weights (sum to 100) (i.e. seating request sizes) are for 2, 4, 6, 8, and 10
    request_size_f = sf.HistogramDist({2:45, 4:30, 6:20, 8:9, 10:1}).generate
    
    # Customer parties loiter for an average of 20 minutes with SD of 10.
    loiter_time_f = sf.NormalDist(20, 10, lower_clip=8, round_digits=0).generate

    wait_time_pricing_env = sf.WaitTimePricingSimEnv(TOTAL_CAPACITY, 
                     base_service_time_f, 
                     base_revenue_f,
                     interarrival_time_f,
                     request_size_f,
                     loiter_time_f,
                     customer_decision_model_generator,
                     agent_action_interval,
                     SIM_DAY_LENGTH,
                     sim_logger,
                     simpy_env,
                     step_report_interval=step_report_interval)
                     
    return wait_time_pricing_env, sim_logger, 14
    

# This function builds a DMV simulation environment which receives customers from 9am to 5pm with 
# varying customer arrival rates at different times throughout the day, with peaks during mid-day. 
# @param agent_action_interval: the number of simulation time minutes representing the time periond in which
#                               the control agent updates premium prices and capacity reservation.
# @returns: a tuple of form (WaitTimePricingSimEnv, BasicSimLog, Operating Hours per Day) 
def dmv_sim_env(simpy_env, agent_action_interval=15, step_report_interval=15):
    SIM_DAY_LENGTH = 60 * 9 # 9am to 6pm = 9 hours (Note: customers stop coming by 5pm).
    AM_11_00 = 60 * 2 
    PM_3_00 = 60 * 7
    PM_5_00 = 60 * 8
    PM_6_00 = 60 * 9

    TOTAL_CAPACITY = 15 # DMV has 15 servers

    sim_logger = sf.BasicSimLog(simpy_env)

    # Customers balk after average of 40 minutes with SD of 15
    balk_threshold_dist = sf.NormalDist(40, 15, lower_clip=15).generate
        
    # Customers reach the maximum willingness to pay for premium after 45 minute waith time with SD of 12 minutes.
    premium_threshold_lower_dist = sf.ConstantDist(0).generate
    premium_upper_lower_delta_dist = sf.NormalDist(45, 12, lower_clip=10).generate
    
    # Customers are willing to pay a maximum of 20 dollars on average to skip the wait with SD of 12
    max_premium_unit_price_dist = sf.NormalDist(20, 12, lower_clip=2).generate
    
    # Customers expect at least 60-90% reduction in waiting time in order to pay for premium
    min_acceptable_wait_reduction_percent_dist = sf.UniformDist(0.6, 0.9, False).generate

    customer_decision_model_generator = sf.CustomerDecisionModelGenerator(balk_threshold_dist,           
                                                    premium_threshold_lower_dist,                   
                                                    premium_upper_lower_delta_dist,                
                                                    max_premium_unit_price_dist,                    
                                                    min_acceptable_wait_reduction_percent_dist)
    customer_decision_model_f = customer_decision_model_generator.generate_decision_f()

    # Base service (i.e. servicing requests) is 15 minutes per seat with SD of 4
    base_service_time_f = lambda x: x * sf.NormalDist(15, 4, lower_clip=3, round_digits=0).generate()
    
    # Base revenue (i.e. service cost) is 10 dollars with SD of 5
    base_revenue_f = lambda x: x * sf.NormalDist(10, 5, lower_clip=3, round_digits=1).generate()
    
    # The mean arrival rates for different time ranges. Peak in mid-day (11:00 am - 3:00 pm).
    mean_arrival_time_phases = [(AM_11_00, sf.ExpInterArrivalTimeDist(25)), 
                                (PM_3_00, sf.ExpInterArrivalTimeDist(55)), 
                                (PM_5_00, sf.ExpInterArrivalTimeDist(10)),
                                (PM_6_00, sf.ConstantDist(60))]
    interarrival_time_f = sf.SimTimeDependentDist(simpy_env, mean_arrival_time_phases).generate
    
    # 1 customer per server, so request sizes are always exactly 1.
    request_size_f = sf.ConstantDist(1).generate
    
    # People don't loiter around DMV for fun, so no loitering time.
    loiter_time_f = sf.ConstantDist(0).generate

    wait_time_pricing_env = sf.WaitTimePricingSimEnv(TOTAL_CAPACITY, 
                     base_service_time_f, 
                     base_revenue_f,
                     interarrival_time_f,
                     request_size_f,
                     loiter_time_f,
                     customer_decision_model_generator,
                     agent_action_interval,
                     SIM_DAY_LENGTH,
                     sim_logger,
                     simpy_env,
                     step_report_interval=step_report_interval)
                     
    return wait_time_pricing_env, sim_logger, 9
    
    
# This function builds an amusement park ride simulation environment which receives riders from 9am to 9pm with 
# varying customer arrival rates at different times throughout the day, with peaks during mid-day. 
# @param agent_action_interval: the number of simulation time minutes representing the time periond in which
#                               the control agent updates premium prices and capacity reservation.
# @returns: a tuple of form (WaitTimePricingSimEnv, BasicSimLog, Operating Hours per Day) 
def amusement_park_ride_sim_env(simpy_env, agent_action_interval=15, step_report_interval=15):
    SIM_DAY_LENGTH = 60 * 14 # 9am to 11pm = 14 hours (Note: customers stop coming by 9pm).
    AM_12_00 = 60 * 3 
    PM_4_00 = 60 * 7
    PM_8_00 = 60 * 11
    PM_9_00 = 60 * 12
    PM_11_00 = SIM_DAY_LENGTH

    TOTAL_CAPACITY = 24 # ride has 24 seats

    sim_logger = sf.BasicSimLog(simpy_env)

    # Customers balk after average of 70 minutes with SD of 15
    balk_threshold_dist = sf.NormalDist(70, 15, lower_clip=15).generate
        
    # Customers reach the maximum willingness to pay for premium after 45 minute waith time with SD of 12 minutes.
    premium_threshold_lower_dist = sf.ConstantDist(0).generate
    premium_upper_lower_delta_dist = sf.NormalDist(45, 12, lower_clip=10).generate
    
    # Customers are willing to pay a maximum of 20 per premium seat with SD of 7
    max_premium_unit_price_dist = sf.NormalDist(20, 7, lower_clip=3).generate
    
    # Customers expect at least 60-90% reduction in waiting time in order to pay for premium
    min_acceptable_wait_reduction_percent_dist = sf.UniformDist(0.6, 0.9, False).generate

    customer_decision_model_generator = sf.CustomerDecisionModelGenerator(balk_threshold_dist,           
                                                    premium_threshold_lower_dist,                   
                                                    premium_upper_lower_delta_dist,                
                                                    max_premium_unit_price_dist,                    
                                                    min_acceptable_wait_reduction_percent_dist)
    customer_decision_model_f = customer_decision_model_generator.generate_decision_f()

    # Base service (i.e. riding time) is a constant of 6 minutes
    base_service_time_f = lambda x: 6
    
    # Base revenue (i.e. cost of riding) is 0 dollars - since park admission is assumed to be already charged.
    base_revenue_f = lambda x: 0
    
    # The mean arrival rates for different time ranges. Peak in mid-day (11:00 am - 3:00 pm).
    mean_arrival_time_phases = [(AM_12_00, sf.ExpInterArrivalTimeDist(45)), 
                                (PM_4_00, sf.ExpInterArrivalTimeDist(100)), 
                                (PM_8_00, sf.ExpInterArrivalTimeDist(250)),
                                (PM_9_00, sf.ExpInterArrivalTimeDist(100)),
                                (PM_11_00, sf.ConstantDist(120))]
    interarrival_time_f = sf.SimTimeDependentDist(simpy_env, mean_arrival_time_phases).generate
    
    # 1 customer per seat, so request sizes are always exactly 1. (Otherwise larger party sizes could go as singles if larger sizes more expensive, singles could pair up if larger sizes offer a discount).
    request_size_f = sf.ConstantDist(1).generate
    
    # Not possible to loiter after the ride, so no loitering time.
    loiter_time_f = sf.ConstantDist(0).generate

    wait_time_pricing_env = sf.WaitTimePricingSimEnv(TOTAL_CAPACITY, 
                     base_service_time_f, 
                     base_revenue_f,
                     interarrival_time_f,
                     request_size_f,
                     loiter_time_f,
                     customer_decision_model_generator,
                     agent_action_interval,
                     SIM_DAY_LENGTH,
                     sim_logger,
                     simpy_env,
                     step_report_interval=step_report_interval)
                     
    return wait_time_pricing_env, sim_logger, 14