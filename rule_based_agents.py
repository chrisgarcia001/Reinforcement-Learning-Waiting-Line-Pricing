# This file contains functions to create rule-base allocation 
# agents and their corresponding control boards. These are used for 
# comparison against RL agents in the experiments.

import sim_framework as sf
from stable_baselines3 import PPO, TD3, SAC

# This provides a corresponding agent for a sf.SimTimeRuleControlBoard object. Since the
# control board is entirely rule-based, this is just a dummy agent that samples the action
# space when predict is called.
class SimTimeRuleAgent:
    def __init__(self, control_board):
        self.control_board = control_board
    
    # Predicts the best action to take (i.e., serves as a RL placeholder).
    # @param state: A gym space
    # @param deterministic: True | False
    # @returns: a pair of form (action, state). Simply samples the action space and returns the origina state unchanged.
    def predict(self, state, deterministic=False):
        return self.control_board.get_action_space().sample(), state

# This constructs a simple agent for the restaurant environment that provides no premium option. Functions as an 
# experimental control/baseline to compare other agents to.
# @param env: An instance of sf.WaitTimePricingSimEnv
# @returns: an (agent, sf.ControlBoardBase) pair. The agent is None, since no additional control beyond the rules is needed.
def build_restaurant_no_premium_agent(env):
    AM_12_00 = 60 * 14
    max_request_size = 10
    price_capacity_rules = [(AM_12_00, (0, 0))]           
    control_board = sf.SimTimeRuleControlBoard(env.simpy_env, max_request_size, price_capacity_rules)
    env.set_control_board(control_board)
    return SimTimeRuleAgent(control_board), control_board


# This constructs a rule-based agent for the restaurant environment that bases capacity allocation pricing on time of day 
# according to expected demand patterns.
# @param env: An instance of sf.WaitTimePricingSimEnv
# @returns: an (agent, sf.ControlBoardBase) pair. The agent is None, since no additional control beyond the rules is needed.
def build_restaurant_rule_agent(env):
    AM_11_30 = 60 * 1.5 
    PM_1_00 = 60 * 3
    PM_6_00 = 60 * 8
    PM_8_00 = 60 * 10
    PM_9_00 = 60 * 11
    AM_12_00 = 60 * 14
    max_request_size = 10
    price_capacity_rules = [(AM_11_30, (1.5, 3)), (PM_1_00, (3.6, 8)), (PM_6_00, (1.5, 3)),
                            (PM_8_00, (7.5, 17)), (PM_9_00, (0.9, 2)), (AM_12_00, (0, 0))]           
    control_board = sf.SimTimeRuleControlBoard(env.simpy_env, max_request_size, price_capacity_rules)
    env.set_control_board(control_board)
    return SimTimeRuleAgent(control_board), control_board
 
# This constructs a simple agent for the DMV environment that provides no premium option. Functions as an 
# experimental control/baseline to compare other agents to.
# @param env: An instance of sf.WaitTimePricingSimEnv
# @returns: an (agent, sf.ControlBoardBase) pair. The agent is None, since no additional control beyond the rules is needed.
def build_dmv_no_premium_agent(env):
    PM_6_00 = 60 * 10
    max_request_size = 1
    price_capacity_rules = [(PM_6_00, (0, 0))]           
    control_board = sf.SimTimeRuleControlBoard(env.simpy_env, max_request_size, price_capacity_rules)
    env.set_control_board(control_board)
    return SimTimeRuleAgent(control_board), control_board


# This constructs a rule-based agent for the restaurant environment that bases capacity allocation pricing on time of day 
# according to expected demand patterns.
# @param env: An instance of sf.WaitTimePricingSimEnv
# @returns: an (agent, sf.ControlBoardBase) pair. The agent is None, since no additional control beyond the rules is needed.
def build_dmv_rule_agent(env):
    AM_11_00 = 60 * 2 
    PM_3_00 = 60 * 7
    PM_5_00 = 60 * 8
    PM_6_00 = 60 * 9
    max_request_size = 1
    price_capacity_rules = [(AM_11_00, (4.55, 2)), (PM_3_00, (10, 5)), (PM_5_00, (1.82, 1)), (PM_6_00, (0, 0))]           
    control_board = sf.SimTimeRuleControlBoard(env.simpy_env, max_request_size, price_capacity_rules)
    env.set_control_board(control_board)
    return SimTimeRuleAgent(control_board), control_board
    
# This constructs a simple agent for the amusement park ride that provides no premium option. Functions as an 
# experimental control/baseline to compare other agents to.
# @param env: An instance of sf.WaitTimePricingSimEnv
# @returns: an (agent, sf.ControlBoardBase) pair. The agent is None, since no additional control beyond the rules is needed.
def build_amusement_park_no_premium_agent(env):
    PM_11_00 = 60 * 14
    max_request_size = 1
    price_capacity_rules = [(PM_11_00, (0, 0))]           
    control_board = sf.SimTimeRuleControlBoard(env.simpy_env, max_request_size, price_capacity_rules)
    env.set_control_board(control_board)
    return SimTimeRuleAgent(control_board), control_board
    
# This constructs a rule-based agent for the restaurant environment that bases capacity allocation pricing on time of day 
# according to expected demand patterns.
# @param env: An instance of sf.WaitTimePricingSimEnv
# @returns: an (agent, sf.ControlBoardBase) pair. The agent is None, since no additional control beyond the rules is needed.
def build_amusement_park_rule_agent(env):
    SIM_DAY_LENGTH = 60 * 14 # 9am to 11pm = 14 hours (Note: customers stop coming by 9pm).
    AM_12_00 = 60 * 3 
    PM_4_00 = 60 * 7
    PM_8_00 = 60 * 11
    PM_9_00 = 60 * 12
    PM_11_00 = SIM_DAY_LENGTH
    max_request_size = 1
    price_capacity_rules = [(AM_12_00, (1.8, 1)), (PM_4_00, (4, 3)), (PM_8_00, (10, 8)), (PM_9_00, (4, 3)), (PM_11_00, (0, 0))]           
    control_board = sf.SimTimeRuleControlBoard(env.simpy_env, max_request_size, price_capacity_rules)
    env.set_control_board(control_board)
    return SimTimeRuleAgent(control_board), control_board