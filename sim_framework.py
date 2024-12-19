import itertools
import random
import simpy
import random as rnd
from scipy.stats import expon, norm, randint, uniform
import gym
import numpy as np

# This class provides a base for simulation logger objects.
class SimLogBase:
    # Abstract logging function.
    # @param info: a dict of form {<attribute>:<value>}
    def log(self, info):
        raise 'SimLogBase.log(info) - Implement Me!'
        
    # Abstract history function.
    # @returns: a list of dicts of form [{<attribute>:<value>}, ...]
    def get_history(self, from_index=0):
        raise 'SimLogBase.get_history(from=0) - Implement Me!'

# This class provides a simple logger primarly to capture measures throughout the
# simulation that will later be used in the analysis.
class BasicSimLog(SimLogBase):
    # Construct an instance of BasicSimLogger.
    # @param env: a simpy.Environmnent instance
    # @param verbose_interval: an int n at which every nth report added will be printed on the console. 
    #                          If None, no reports are printed.
    def __init__(self, env, verbose_inteval=None):
        self.history = [] # This is what holds all collected measures.
        self.env = env
        self.verbose_inteval = verbose_inteval
    
    # Logging function implementation.
    # @param info: a dict of form {<attribute>:<value>}
    def log(self, info):
        if self.verbose_inteval != None and len(self.history) % self.verbose_inteval == 0:
            s = 'time: ' + str(self.env.now)
            for k in sorted(info.keys()):
                s += '; ' + str(k) + ': ' + str(info[k])
            print(s)
        info['time'] = self.env.now
        self.history.append(info)
    
    # Gets the history for this log.
    # @returns: a list of dicts of form [{<attribute>:<value>}, ...]
    def get_history(self, from_index=0):
        return self.history[from_index:]
        

# -------------------- Distributions --------------------------------

# Base class for distributions.
class DistributionBase:
    def generate(self):
        raise 'DistributionBase.generate_value() - Implement me!'

# This class simply always returns a constant. Primarily used as a sub-component in others.
class ConstantDist(DistributionBase):
    def __init__(self, constant_value):
        self.val = constant_value
    
    # Return the constant value
    def generate(self):
        return self.val
        
    def __str__(self):
        return 'ConstantDist(' + str(self.val) + ')'
        
# This class generates exponentially distributed interarrival times in minutes.
# Notes on using expon: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html        
class ExpInterArrivalTimeDist(DistributionBase):
    # @param arrivals_per_hour: hourly arrival rate
    def __init__(self, arrivals_per_hour):
        self.arrivals_per_hour = arrivals_per_hour # This is lambda in queuing theoretic notation
    
    # Generate a random value according to the distribution. NOTE: This will generate arrival times in MINUTES.
    def generate(self):
        return expon(scale=1 / max(1, self.arrivals_per_hour)).rvs() * 60


# This class generates normally distributed values.
# Notes on using norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
class NormalDist(DistributionBase):
        # @param mu: the mean
        # @param sigma: the standard deviation
        # @param lower_clip: the lowest value that can be generated
        # @param upper_clip: the highest value that can be generated
        def __init__(self, mu, sigma, lower_clip=None, upper_clip=None, round_digits=None):
            self.mu = mu
            self.sigma = sigma
            self.lower_clip = lower_clip if round_digits == None or lower_clip == None else round(lower_clip, round_digits)
            self.upper_clip = upper_clip if round_digits == None or upper_clip == None else round(upper_clip, round_digits)
            self.round_digits = round_digits
                
            
        # Generate a random value according to the distribution.
        def generate(self):
            val = norm(self.mu, self.sigma).rvs()
            if self.lower_clip != None and val < self.lower_clip:
                return self.lower_clip 
            elif self.upper_clip != None and val > self.upper_clip:
                return self.upper_clip 
            else:
                return val if self.round_digits == None else round(val, self.round_digits)

# This class generates uniformly distributed values.
class UniformDist(DistributionBase):
    # @param lb: the lower bound
    # @param ub: the upper bound
    # @param discrete = True | False = whether or not the distribution is discrete
    def __init__(self, lb=0, ub=1, discrete=False):
        self.lb = lb
        self.ub = ub + 1 if discrete else ub
        self.discrete = discrete
        
    # Generate a random value according to the distribution.
    def generate(self):
        if self.discrete:
            return randint(self.lb, self.ub).rvs()
        else:
            return (uniform().rvs() * (self.ub - self.lb)) + self.lb
            
 # This constructs a distribution from a histogram.
class HistogramDist(DistributionBase):
    # Construct a HistogramDist object
    # @param values_with_weights: a dict of form {<value>:<weight>, ...}. Weights can correspond
    #                             to probabilities, or general weights. If probabilities, they should
    #                             sum to 1.
    def __init__(self, values_with_weights):
        self.total_weight = 0
        self.ordered_items = []
        items = values_with_weights.items()
        for value, weight in sorted(items, key=lambda x:x[0]):
            self.total_weight += weight
            self.ordered_items.append((self.total_weight, value))
        self.uniform_gen = UniformDist(lb=0, ub=self.total_weight, discrete=False)
    
    # Generate a random value according to the distribution.
    def generate(self):
        v = self.uniform_gen.generate()
        for cumulative_weight, value in self.ordered_items:
            if v <= cumulative_weight:
                return value
        raise 'HistogramDist.generate() - Out of bounds error!'
        
    
 
# This class is a composite distribution that uses different distributions according to 
# the current simulation time.
class SimTimeDependentDist(DistributionBase):
    # @param: simpy_env: a Simpy Environment instance
    # @param phases: A list of form[(<Cutoff Time>, <DistributionBase subclass>), ...].
    #                Starting with the first pair, if time <= cutoff_time then the corresponding
    #                distribution is used.
    def __init__(self, simpy_env, phases):
        self.env = simpy_env
        self.phases = sorted(phases, key=lambda x:x[0])
        self.cycle_length = self.phases[len(self.phases) - 1][0]
    
    # Generate a random value according from the distribution that corresponds
    # to the current simulation time.
    def generate(self):
        for cutoff_time, distribution in self.phases:
            if int(self.env.now) % (self.cycle_length + 1) <= cutoff_time:
                return distribution.generate()
        raise BaseException('SimTimeDependentDist.generate() - Out of bounds error (cutoff_time = ' + str(cutoff_time) + 
                            ', cycle_length = ' + str(self.cycle_length) + ', modulo = ' + 
                            str(self.env.now % (self.cycle_length + 1)) + ')')
            
# -------------------- End Distributions ----------------------------


# -------------------- Customer Decision Model Generator ------------

# This class models the customer decision process and generates random customer decision functions
# from a set of user-specified distributions. Generated decision functions have the following form:
#     f(<regular wait time>, <premium wait time>, <premium price>, <request size>) -> 'balk' | 'regular' | 'premium'
class CustomerDecisionModelGenerator:
    # Constructs a CustomerDecisionModelGenerator instance.
    # @param balk_threshold_dist: a function of form f() -> minutes. Generates the longest waiting time before balking.
    # @param premium_threshold_lower_dist: a function of form f() -> minutes. Generates the waiting time threshold 
    #                                      below which the customer will not pay any amount for premium option (called Tpl).
    # @param premium_upper_lower_delta_dist: a function of form f() -> minutes. Let Tpu represent the waiting time threshold 
    #                                        at or above which the customer will pay their maximum amount (assuming no 
    #                                        wait for premium). Then Tpu = Tpl + delta. This function generates delta.
    # @param max_premium_unit_price_dist: a function of form f() -> dollars. Generates the maximum premium price per premium
    #                                     unit a customer will pay.
    # @param min_acceptable_wait_reduction_percent_dist: a function of form f() -> [0, 1]. 
    def __init__(self,
                 balk_threshold_dist,                            # minutes
                 premium_threshold_lower_dist,                   # minutes
                 premium_upper_lower_delta_dist,                 # minutes
                 max_premium_unit_price_dist,                    # dollars
                 min_acceptable_wait_reduction_percent_dist):    # [0, 1]
        self.Tb_dist = balk_threshold_dist
        self.Tpl_dist = premium_threshold_lower_dist
        self.Tpu_delta_dist = premium_upper_lower_delta_dist
        self.Tm_dist = max_premium_unit_price_dist
        self.Twr_dist = min_acceptable_wait_reduction_percent_dist
    
    # Generates a random customer decision model function.
    # @returns: a deterministic function of form f(<regular wait time>, <premium wait time>, <premium price>, <request size>) -> 'balk' | 'regular' | 'premium'
    def generate_decision_f(self):
        Tb = self.Tb_dist()                # The balk threshold for this customer (in minutes)
        Tpl = self.Tpl_dist()              # The lower premium threshold for this customer (in minutes)
        Tpu = Tpl + self.Tpu_delta_dist()  # The upper premium threshold for this customer (in minutes)
        Tm = self.Tm_dist()                # The absolute maximum price (per seat/unit) this customer will pay for premium (in dollars)
        Twr = self.Twr_dist()              # The minimum wait reduction (%) achieved by premium that this customer will pay for (in [0, 1])
        
        # This function returns an element from ['balk', 'regular', 'premium']
        # @param Wr = regular wait time estimate
        # @param Wp = premium wait time estimate
        # @param Q = premium price quote (for total requested size)
        # @param S = request size in units
        def decision_f(Wr, Wp, Q, S):
            if Wr == 0:
                return 'regular'
            Rpr = Wp / max(1, Wr)
            max_accept_price = Tm * S * ((min(Tpu, Wr) - Tpl) / (Tpu - Tpl)) * (1 - Rpr)
            if Q <= max_accept_price and Tpl < Wr and 1 - Rpr >= Twr:
                return 'premium'
            elif Wr > Tb:
                return 'balk'
            else:
                return 'regular'
        return decision_f

# -------------------- End Customer Decision Model Generator --------



# -------------------- Control Board Classes ------------------------

# This class abstracts the controller agents inside the simulation framework. Each agent
# will continuously adjust two parameters in order to optimize the revenue:
#  1) the price for premium
#  2) the level of premium capacity reserved
# This class allows the simulation environment to know at any time the current
# premium price (in dollars) for a given request size and how much capacity  
# should be allocated to premium (in units/seats).
class ControlBoardBase:
    # Gets the current premium price
    # @param request_size: the number of units/seats requested
    # @returns: the price quote in dollars
    def premium_price(self, request_size):
        raise BaseException('ControlBoardBase.premium_price(request_size) is abstract - Implement me!')
    
    # Gets the current capacity (in units/seats) to be allocated to premium
    # @returns: the number of units/seats.
    def premium_capacity_level(self):
        raise BaseException('ControlBoardBase.premium_capacity_level(request_size) is abstract - Implement me!')
    
    # This translates an action object from an agent into the current premium price and capacity level.
    # @param action: an action that matches the action space.
    def update(self, action):
        raise BaseException('ControlBoardBase.update(action) is abstract - Implement me!')
    
    # This constructs/returns an instance of a gym.spaces object
    def get_action_space(self):
        raise BaseException('ControlBoardBase.get_action_space() is abstract - Implement me!')
    
    # This snaps the request size to the appropriate value.
    def snap_to_valid_request_size(self, request_size):
        raise BaseException('ControlBoardBase.snap_to_valid_request_size(request_size) is abstract - Implement me!')
    
    # This constructs a string representation of this object    
    def __str__(self):
        raise BaseException('ControlBoardBase.str() is abstract - Implement me!')
        
# This class implements a basic agent that uses simple rules to adjust the premium capacity reserve and
# per-unit premium pricing based on the simulation time.
class SimTimeRuleControlBoard(ControlBoardBase):
    # @param simpy_env: A Simpy Environment instance
    # @param max_request_size: The maximum premium request size allowed
    # @param rules: A list of form[(<Cutoff Time>, (<Premium Unit Price>, <Premium Capacity Reserve>)), ...].
    #               Starting with the first pair, if time <= cutoff_time then the corresponding
    #               (<Premium Unit Price>, <Premium Capacity Reserve>) pair is used.
    def __init__(self, simpy_env, max_request_size, rules):
        self.request_sizes = list(range(1, max_request_size + 1))
        self.dist = SimTimeDependentDist(simpy_env, [(time, ConstantDist(values)) for time, values in rules])
    
    # Gets the current premium price
    # @param request_size: the number of units/seats requested
    # @returns: the price quote in dollars
    def premium_price(self, request_size):
        if request_size > self.request_sizes[len(self.request_sizes) - 1] or self.premium_capacity_level() == 0:
            return float('infinity')
        rounded_size = self.snap_to_valid_request_size(request_size)
        unit_price, _ = self.dist.generate()
        return rounded_size * unit_price
        
    # Gets the current capacity (in units/seats) to be allocated to premium
    # @returns: the number of units/seats
    def premium_capacity_level(self):
        _, reserve = self.dist.generate()
        return reserve
    
    # Since this class only uses the simulation time to determine how to set the parameters,
    # nothing needs to be done here.
    # @param action: an action that matches the action space.
    def update(self, action):
        pass    
        
    # Constructs/returns a dummy instance of a gym.spaces object (since this class does not respond to actions).
    def get_action_space(self):
        return gym.spaces.Box(low=np.float32([-1]), 
                              high=np.float32([1]))
    
    # This snaps a party request size to the appropriate valid request size.
    def snap_to_valid_request_size(self, request_size):
        for s in self.request_sizes:
            if s >= request_size:
                return s
        return request_size
        
    # This constructs a string representation of this object.
    def __str__(self):
        s = 'SimTimeRuleControlBoard (request size, price): ' + str(list(zip(self.request_sizes, [self.premium_price(x) for x in self.request_sizes])))
        return s


# This class translates a Box action into internal simulation control parameters (premium prices and capacity reservation level).
# In this representation, the agent sets a price for each unique valid request size individually.
class RLMultiPriceBoxActionTranslator(ControlBoardBase):
    # @param total_capacity: the total number of capacity units available_capacity
    # @param min_reserve_capacity: the minimum reservable capacity for premium. Note that 0 <= min_reserve_capacity <= max_reserve_capacity.
    # @param max_reserve_capacity: the maximum reservable capacity for premium. Note that max_reserve_capacity <= total_capacity.
    # @param max_request_size: The maximum premium request size allowed
    # @param max_price: the maximum price that can be charged for a request of any size.
    # @param min_price: the minimum price that can be charged for a request of any size.
    def __init__(self, total_capacity, min_reserve_capacity,
                 max_reserve_capacity, max_request_size, max_price, min_price=0):
        assert max_reserve_capacity <= total_capacity
        self.total_capacity = total_capacity
        self.min_reserve_capacity = min_reserve_capacity
        self.max_reserve_capacity = max_reserve_capacity
        self.request_sizes = list(range(1, max_request_size + 1))
        self.max_price = max_price
        self.min_price = min_price
        self.current_prices = np.zeros(len(self.request_sizes))
        self.current_premium_reserve = 0
    
    # Gets the current premium price
    # @param request_size: the number of units/seats requested
    # @returns: the price quote in dollars
    def premium_price(self, request_size):
        for i, size in enumerate(self.request_sizes):
            if request_size <= size and self.current_premium_reserve > 0:
                return self.current_prices[i]
        return float('infinity')
    
    # Gets the current capacity (in units/seats) to be allocated to premium
    # @returns: the number of units/seats.
    def premium_capacity_level(self):
        return self.current_premium_reserve
    
    # This translates an action object from an agent into the current premium price and capacity level.
    # @param action: a Box action that matches the action space.
    def update(self, action):
        tf = lambda x, min_fx, max_fx, digits: round(min_fx + (((x + 1.0) / 2.0) * (max_fx - min_fx)), digits)
        self.current_prices = [tf(x, self.min_price, self.max_price, 2) for x in action[:len(action) - 1]] 
        # Ensure no larger request size is priced lower than a smaller request size - necessary for consistency & fairness.
        for i in range(1, len(self.current_prices)):
            self.current_prices[i] = max(self.current_prices[i], self.current_prices[i-1])
        self.current_premium_reserve = tf(action[len(action) - 1], self.min_reserve_capacity, self.max_reserve_capacity, 0)
        
    # Construct the action space from the parameters.    
    def get_action_space(self):
        # Per the StableBaselines3 recommendations, use a symmetric box ranging from -1 to 1. The update function above maps to appropriate values.
        # Action Space = Box(
        #                    <smallest request size price>,        #(-1, 1) which maps to (0, self.max_price) 
        #                    <second smallest req. size price>,    #(-1, 1) which maps to (0, self.max_price)   
        #                    ...,
        #                    <largest req. size>,                  #(-1, 1) which maps to (0, self.max_price) 
        #                    <premium capacity reserve>            #(-1, 1) which maps to (0, self.max_reserve_capacity)
        return gym.spaces.Box(low=np.float32([-1 for i in range(len(self.request_sizes) + 1)]),
                              high=np.float32([1 for i in range(len(self.request_sizes) + 1)]))
                                              
    # This snaps a party request size to the appropriate valid request size.
    def snap_to_valid_request_size(self, request_size):
        for s in self.request_sizes:
            if s >= request_size:
                return s
        return request_size
    
    # This constructs a string representation of this object.
    def __str__(self):
        s = 'RLMultiPriceBoxActionTranslator (request size, price): ' + str(list(zip(self.request_sizes, [self.premium_price(x) for x in self.request_sizes])))
        return s


# This class translates a Box action into internal simulation control parameters (premium price/unit and capacity reservation level).
# In this representation, the agent sets a single price per unit. The price of each valid request size is then the number of units requested
# multiplied by this price. This gives a simple model with a comparatively small search space.
class RLSinglePriceBoxActionTranslator(ControlBoardBase):
    # @param total_capacity: the total number of capacity units available_capacity
    # @param min_reserve_capacity: the minimum reservable capacity for premium. Note that 0 <= min_reserve_capacity <= max_reserve_capacity.
    # @param max_reserve_capacity: the maximum reservable capacity for premium. Note that max_reserve_capacity <= total_capacity.
    # @param max_request_size: The maximum premium request size allowed
    # @param max_price: the maximum price per unit requested that can be charged.
    # @param min_price: the minimum price per unit requested that can be charged.
    def __init__(self, total_capacity, min_reserve_capacity,
                 max_reserve_capacity, max_request_size, max_price, min_price=0):
        assert max_reserve_capacity <= total_capacity
        self.total_capacity = total_capacity
        self.min_reserve_capacity = min_reserve_capacity
        self.max_reserve_capacity = max_reserve_capacity
        self.request_sizes = list(range(1, max_request_size + 1))
        self.min_price = min_price
        self.max_price = max_price
        self.current_prices = np.zeros(len(self.request_sizes))
        self.current_premium_reserve = 0
    
    # Gets the current premium price
    # @param request_size: the number of units/seats requested
    # @returns: the price quote in dollars
    def premium_price(self, request_size):
        for i, size in enumerate(self.request_sizes):
            if request_size <= size and self.current_premium_reserve > 0:
                return self.current_prices[i]
        return float('infinity')
    
    # Gets the current capacity (in units/seats) to be allocated to premium
    # @returns: the number of units/seats.
    def premium_capacity_level(self):
        return self.current_premium_reserve
    
    # This translates an action object from an agent into the current premium price and capacity level.
    # @param action: a Box action that matches the action space.
    def update(self, action):
        tf = lambda x, min_fx, max_fx, digits: round(min_fx + (((x + 1.0) / 2.0) * (max_fx - min_fx)), digits)
        unit_price = tf(action[0], self.min_price, self.max_price, 2)
        self.current_prices = [unit_price * x for x in self.request_sizes] 
        self.current_premium_reserve = tf(action[1], self.min_reserve_capacity, self.max_reserve_capacity, 0)
        
    # Construct the action space from the parameters.    
    def get_action_space(self):
        # Per the StableBaselines3 recommendations, use a symmetric box ranging from -1 to 1. The update function above maps to appropriate values.
        # Action Space = Box(<price per unit>,               #(-1, 1) which maps to (0, self.max_price) 
        #                    <premium capacity reserve>)     #(-1, 1) which maps to (0, self.max_reserve_capacity)
        return gym.spaces.Box(low=np.float32([-1, -1]), high=np.float32([1, 1]))
                                              
    # This snaps a party request size to the appropriate valid request size.
    def snap_to_valid_request_size(self, request_size):
        for s in self.request_sizes:
            if s >= request_size:
                return s
        return request_size
    
    # This constructs a string representation of this object.
    def __str__(self):
        s = 'RLSinglePriceBoxActionTranslator (request size, price): ' + str(list(zip(self.request_sizes, [self.premium_price(x) for x in self.request_sizes])))
        return s
 

    
# -------------------- End Control Board Classes --------------------

class Party:
    # @param decision_f: a deterministic function of form f(<regular wait time>, <premium wait time>, <premium price>, <request size>) -> 'balk' | 'regular' | 'premium'
    def __init__(self, 
                 name, 
                 request_size, 
                 loiter_time, 
                 decision_f):
        self.name = name
        self.request_size = request_size
        self.loiter_time = loiter_time
        self.decision_f = decision_f
        
    def __str__(self):
        return '(' + str(self.name) + ', ' + str(self.request_size) + ')'
    
    def balk(self, reg_wait_time, prem_wait_time, prem_price):
        return self.decision_f(reg_wait_time, prem_wait_time, prem_price, self.request_size) == 'balk'
    
    def premium(self, reg_wait_time, prem_wait_time, prem_price):
        return self.decision_f(reg_wait_time, prem_wait_time, prem_price, self.request_size) == 'premium'
        
    def decision(self, reg_wait_time, prem_wait_time, prem_price):
        return self.decision_f(reg_wait_time, prem_wait_time, prem_price, self.request_size)

        
class TwoClassService:
    # base_service_time_f: A function of form f(<request size>) -> minutes. Models providing service excluding loitering time.
    # base_revenue_f: A function of form f(<request size>) -> dollars. Models revenue for basic service (excluding wait skipping).
    def __init__(self, 
                 env, 
                 capacity, 
                 base_service_time_f, 
                 base_revenue_f,
                 control_board,
                 sim_logger):
        self.env = env
        self.capacity = capacity
        self.regular = simpy.Container(env, capacity, init=capacity)
        self.premium = simpy.Container(env, capacity, init=0)
        self.reserve_level = 0 
        self.base_service_time_f = base_service_time_f
        self.base_revenue_f = base_revenue_f
        self.control_board = control_board
        self.logger = sim_logger
        self.regular_queue_length = 0
        self.premium_queue_length = 0
        
        
    def update_premium_capacity_reserve(self):
        level = self.control_board.premium_capacity_level()
        self.logger.log({'event':'update_premium_capacity', 'level':level}) 
        if level - self.reserve_level > 0:
            diff = level - self.reserve_level
            self.reserve_level = level
            yield self.regular.get(diff)
            yield self.premium.put(diff)
        elif level - self.reserve_level < 0 and self.premium_queue_length == 0:
            diff = self.reserve_level - level
            self.reserve_level = level
            yield self.premium.get(diff)
            yield self.regular.put(diff)
    
    def regular_capacity_in_use(self):
        return self.capacity - self.reserve_level - self.regular.level
        
    def premium_capacity_in_use(self):
        return self.reserve_level - self.premium.level
            
    def update_queue_length(self, request_size, premium):
        if premium:
            self.premium_queue_length += request_size
        else:
            self.regular_queue_length += request_size
    
    def current_wait_time_estimate(self, request_size, premium):
        queue_length = self.premium_queue_length if premium else self.regular_queue_length
        return queue_length + request_size
            
          
    def serve(self, party):
        wr = self.current_wait_time_estimate(party.request_size, False)
        wp = self.current_wait_time_estimate(party.request_size, True)
        pp = self.control_board.premium_price(party.request_size) 
        rq = self.regular_queue_length
        pq = self.premium_queue_length
        reserve_level_at_offer = self.reserve_level
        self.logger.log({'event':'customer_decision', 'party':party, 'request_size':party.request_size, 'regular_wait_estimate':wr,
                         'decision':party.decision(wr, wp, pp), 'premium_wait_estimate':wp, 'premium_price_quote':pp,
                         'regular_queue_length':rq, 'premium_queue_length':pq, 
                         'regular_capacity_in_use':self.regular_capacity_in_use(),
                         'premium_capacity_in_use':self.premium_capacity_in_use(),
                         'premium_reserve':self.reserve_level})
        if not(party.balk(wr, wp, pp)):
            premium = party.premium(wr, wp, pp) and party.request_size <= self.reserve_level
            prem_rev = pp if premium else 0
            self.update_queue_length(party.request_size, premium)
            container = self.premium if premium else self.regular
            queue_entry_time = self.env.now
            yield container.get(party.request_size)
            self.update_queue_length(-party.request_size, premium)
            service_start_time = self.env.now
            self.logger.log({'event':'service_start', 'party':party, 'premium':premium})
            base_service_time = self.base_service_time_f(party.request_size)
            base_rev = self.base_revenue_f(party.request_size)
            total_service_time = base_service_time + party.loiter_time
            yield self.env.timeout(total_service_time) 
            yield container.put(party.request_size)
            self.logger.log({'event':'service_complete', 'party':party, 'premium':premium,
                             'base_service_time':base_service_time, 'base_revenue':base_rev, 
                             'premium_revenue':prem_rev, 'total_revenue':base_rev + prem_rev, 
                             'loiter_time':party.loiter_time, 'total_service_time':total_service_time,
                             'total_time':self.env.now - queue_entry_time})
        
        
class WaitTimePricingSimEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, 
                 total_capacity, 
                 base_service_time_f, 
                 base_revenue_f,
                 interarrival_time_f,
                 request_size_f,
                 loiter_time_f,
                 customer_decision_model_generator,
                 agent_action_interval,
                 sim_day_length,
                 sim_logger,
                 simpy_env=simpy.Environment(),
                 control_board=None,
                 step_report_interval=None):
        super(WaitTimePricingSimEnv, self).__init__()
        self.total_capacity = total_capacity
        self.base_service_time_f = base_service_time_f
        self.base_revenue_f = base_revenue_f
        self.interarrival_time_f = interarrival_time_f
        self.request_size_f = request_size_f
        self.loiter_time_f = loiter_time_f
        self.customer_decision_model_generator = customer_decision_model_generator
        self.agent_action_interval = agent_action_interval
        self.day_length = sim_day_length # In minutes
        self.current_day = -1 
        self.logger = sim_logger
        self.simpy_env = simpy_env
        self.control_board = control_board
        self.step_report_interval = step_report_interval
        self.steps = 0
        
        self.state_reward_history = [] # [(observation, reward), ...]
        
        self.action_space = self.control_board.get_action_space() if self.control_board != None else None
        self.two_class_service = None
        self.most_recent_history_index = 0
        self.MAX_RESOURCE_UNITS = 10 * self.total_capacity
        
        # Observation Space = Box(<current regular queue length>,                                        #(0, 10 * self.MAX_RESOURCE_UNITS)
        #                         <current premium queue length>,                                        #(0, 10 * self.MAX_RESOURCE_UNITS)
        #                         <balk count since last action>,                                        #(0, 10 * self.MAX_RESOURCE_UNITS)
        #                         <premium accept count since last action>,                              #(0, 10 * self.MAX_RESOURCE_UNITS)
        #                         <regular accept count since last action>                               #(0, 10 * self.MAX_RESOURCE_UNITS)
        #                         <current regular capacity in use>                                      #(-self.total_capacity, self.total_capacity) 
        #                         <current premium capacity in use>                                      #(-self.total_capacity, self.total_capacity)
        #                         <avg. regular wait time estimate (queue length + party size as proxy)> #(0, 1000 * self.MAX_RESOURCE_UNITS)
        #                         <avg. premium wait time estimate (queue length + party size as proxy)> #(0, 1000 * self.MAX_RESOURCE_UNITS)
        self.observation_space = gym.spaces.Box(low=np.float32([0 for i in range(5)] + 
                                                               [-self.total_capacity, -self.total_capacity] + 
                                                               [0 for i in range(2)]),
                                                high=np.float32([10 * self.MAX_RESOURCE_UNITS for i in range(5)] +
                                                                [self.total_capacity for i in range(2)] +
                                                                [1000 * self.MAX_RESOURCE_UNITS for i in range(2)]))
    
    # Allows the internal control board to be set from the outside.
    # @param control_board: an instance of ControlBoardBase
    def set_control_board(self, control_board):
        self.control_board = control_board
        self.action_space = self.control_board.get_action_space()
    
    # This process generates a continual stream of customers/parties as 
    # determined by the constructor parameters.
    def generate_customers(self):
        for i in itertools.count():
            t = self.interarrival_time_f()
            yield self.simpy_env.timeout(t)
            p = Party('Party_' + str(i), self.request_size_f(), self.loiter_time_f(), 
                      self.customer_decision_model_generator.generate_decision_f()) 
            self.simpy_env.process(self.two_class_service.serve(p))
    
    # This returns a state, reward pair for the current state.
    # @param from_index: the history index to start with.
    def get_current_state_and_reward(self, from_index=None):
        if from_index == None:
            from_index = self.most_recent_history_index
        if self.two_class_service == None:
            return np.zeros(8), 0
        history = self.logger.get_history(from_index=from_index)
        balk, prem, reg, reg_wait_est, prem_wait_est, revenue = 0, 0, 0, 0, 0, 0
        for h in history:
            if 'event' in h:
                if h['event'] == 'customer_decision':
                    reg_wait_est += h['regular_wait_estimate']
                    prem_wait_est += h['premium_wait_estimate']
                    if h['decision'] == 'balk':
                        balk += 1
                    elif h['decision'] == 'premium':
                        prem += 1
                    else:
                        reg += 1
            if 'total_revenue' in h:
                revenue += h['total_revenue']
        reg_wait_est /= max(1, len(history))
        prem_wait_est /= max(1, len(history))
        state = np.float32([self.two_class_service.regular_queue_length,
                            self.two_class_service.premium_queue_length,
                            balk, 
                            prem, 
                            reg, 
                            self.two_class_service.regular_capacity_in_use(),
                            self.two_class_service.premium_capacity_in_use(),
                            reg_wait_est,
                            prem_wait_est])
        return state, revenue
    
    
    # Execute an action in the environment and get the resulting feedback.
    def step(self, action):
        if self.control_board == None:
            raise BaseException('Error in WaitTimePricingSimEnv(gym.Env).step: self.control_board is None - please set this parameter before attempting to step.')
        self.control_board.update(action)
        self.simpy_env.process(self.two_class_service.update_premium_capacity_reserve())
        self.simpy_env.run(until=self.simpy_env.now + self.agent_action_interval)
        if self.step_report_interval != None and self.steps % self.step_report_interval == 0:
            self.render()
        observation, reward = self.get_current_state_and_reward()
        self.state_reward_history.append((observation, reward))
        self.most_recent_history_index = len(self.logger.get_history())
        done = self.simpy_env.now >= (self.current_day + 1) * self.day_length
        info = {}
        self.steps += 1
        return observation, reward, done, info
    
    # Reset the simulation two class service and simulation environment.
    def reset(self):
        self.most_recent_history_index = len(self.logger.get_history())
        while self.simpy_env.now >= (self.current_day + 1) * self.day_length:
            self.current_day += 1
        if self.two_class_service == None:
            self.two_class_service = TwoClassService(self.simpy_env,
                                                     self.total_capacity,
                                                     self.base_service_time_f,
                                                     self.base_revenue_f,
                                                     self.control_board,
                                                     self.logger)
            self.simpy_env.process(self.generate_customers())
        elif self.two_class_service.control_board == None:
            self.two_class_service.control_board = self.control_board
        observation, _ = self.get_current_state_and_reward()
        return observation  # reward, done, info can't be included
    
    # Provide a rendering of the current state on the command line.
    def render(self, mode="human"):
        ob, revenue = self.get_current_state_and_reward()
        print("\n-------------------------------------------")
        print('Current State (Total History Length =', str(len(self.logger.get_history())) + '):')
        print('  Current Step:', self.steps)
        print('  Current Simulation Time:', self.simpy_env.now)
        print('  Regular Queue Length:', ob[0])
        print('  Premium Queue Length:', ob[1])
        print('  Premium Reserve Level:', self.two_class_service.reserve_level)
        print('  Regular Level:', self.two_class_service.regular.level)
        print('  Premium Level:', self.two_class_service.premium.level)
        print('  Regular Capacity in Use:', self.two_class_service.regular_capacity_in_use())
        print('  Premium Capacity in Use:', self.two_class_service.premium_capacity_in_use())
        print('  Control Board:', str(self.control_board))
        print('  Balk Count (Since Last Action):', ob[2])
        print('  Premium Count (Since Last Action):', ob[3])
        print('  Regular Count (Since Last Action):', ob[4])
        print('  Regular Wait Time Estimate Avg. (Since Last Action):', ob[7])
        print('  Premium Wait Time Estimate Avg. (Since Last Action):', ob[8])
        print('  Total Revenue (Since Last Action):', revenue)
        
    # Provide a dummy close method (none needed in this environment).
    def close (self):
        pass  


