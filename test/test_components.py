import sys
sys.path.insert(1, '../')

import sim_framework as sf

# ----------- Test Distributions ------------------------

# Constant distribution test
def test_constant_dist():
    c = sf.ConstantDist(5)
    assert(c.generate() == 5)
    print('Constant Distribution Test Passed')

# Discrete uniform distribution test
def test_discrete_uniform_dist():
    u1 = sf.UniformDist(5, 10, True)
    for i in range(50):
        value = u1.generate() 
        assert(value >= 5 and value <= 10)
    print('Uniform Distribution Test (discrete) Passed')

# Continuous uniform distrubution test
def test_continuous_uniform_dist():
    u2 = sf.UniformDist(5, 10, False)
    for i in range(50):
        value = u2.generate() 
        assert(value > 5 and value <=10)
    print('Uniform Distribution Test (continuous) Passed')   

# Exponential hourly interarrival time distribution test
def test_exp_interarrival_dist():
    arrivals_per_hour = 45
    n_hours = 50
    ep = sf.ExpInterArrivalTimeDist(arrivals_per_hour)
    ep_total = 0
    for i in range(arrivals_per_hour * n_hours):
        ep_total += ep.generate()
    assert(ep_total / (arrivals_per_hour * n_hours) >= 0.9 * (60 / arrivals_per_hour))
    assert(ep_total / (arrivals_per_hour * n_hours) <= 1.1 * (60 / arrivals_per_hour))
    print(sf.ExpInterArrivalTimeDist(0).generate())
    print('Exponential Hourly Interarrival Time Distribution Test Passed')   

# Uncensored normal distribution test
def test_uncensored_normal_dist():
    mu = 10
    sigma = 2.5
    nd1 = sf.NormalDist(mu, sigma)
    nd1_total = 0
    nd1_iterations = 1000
    nd1_avg = sum([nd1.generate() for i in range(nd1_iterations)]) / nd1_iterations
    assert(nd1_avg >= 0.9 * mu)
    assert(nd1_avg <= 1.1 * mu)
    print('Normal Distribution Test (uncensored) Passed')  

# Censored normal distribution test
def test_censored_normal_dist():
    mu = 10
    sigma = 2.5
    low_clip = 5.0
    high_clip = 15.0
    nd2 = sf.NormalDist(10, 2.5, lower_clip=low_clip, upper_clip=high_clip)
    nd2_iterations = 1000
    nd2_total = 0
    for i in range(nd2_iterations):
        value = nd2.generate() 
        nd2_total += value
        assert(value >= low_clip)
        assert(value <= high_clip)
    nd2_avg = nd2_total / nd2_iterations
    assert(nd2_avg  >= 0.9 * mu)
    assert(nd2_avg <= 1.1 * mu)    
    print('Normal Distribution Test (censored) Passed') 

# Histogram distribution test
def test_histogram_dist():
    w1 = 10
    w2 = 20
    w3 = 30
    weighted_hist_values = {3:w3, 2:w2, 1:w1}
    hd = sf.HistogramDist(weighted_hist_values)
    hd_iterations = 2000
    hd_data = [hd.generate() for i in range(hd_iterations)]
    ones = [x for x in hd_data if x == 1]
    twos = [x for x in hd_data if x == 2]
    threes = [x for x in hd_data if x == 3]
    assert(len(ones) >= 0.9 * (w1 / sum([w1, w2, w3])) * hd_iterations)
    assert(len(ones) <= 1.1 * (w1 / sum([w1, w2, w3])) * hd_iterations)
    assert(len(twos) >= 0.9 * (w2 / sum([w1, w2, w3])) * hd_iterations)
    assert(len(twos) <= 1.1 * (w2 / sum([w1, w2, w3])) * hd_iterations)
    assert(len(threes) >= 0.9 * (w3 / sum([w1, w2, w3])) * hd_iterations)
    assert(len(threes) <= 1.1 * (w3 / sum([w1, w2, w3])) * hd_iterations)
    print('Histogram Distribution Test Passed') 

# Test the simulation time-dependent distribution.
def test_sim_time_dependent_dist():
    class MockSimEnv:
        def __init__(self, now=0):
            self.now = now

    mu1, mu2, mu3 = 5.0, 10.0, 15.0
    smd1, smd2, smd3 = sf.NormalDist(mu1, 1.0), sf.NormalDist(mu2, 1.0), sf.NormalDist(mu3, 1.0) 
    t1, t2, t3, t4, t5, t6 = 5, 10, 15, 20, 25, 30
    sim_env = MockSimEnv()
    phases = [(t1, smd1), (t2, smd2), (t3, smd3)]
    stdd = sf.SimTimeDependentDist(sim_env, phases)
    stdd_phase_iterations = 1000
    for t, mu in [(t1, mu1), (t2, mu2), (t3, mu3), (t4, mu1), (t5, mu2), (t6, mu3)]:
        sim_env.now = t - 1
        v = sum([stdd.generate() for i in range(stdd_phase_iterations)]) / stdd_phase_iterations
        assert(0.9 * mu <= v <= 1.1 * mu)
    print('Simulation Time-Dependent Distribution Test Passed')     

# ----------- Test Customer Choice Decision Model -------

def test_customer_decision_model_generator():
    cdmg = sf.CustomerDecisionModelGenerator(sf.ConstantDist(60).generate,  # Balk after 60 minutes
                                             sf.ConstantDist(5).generate,   # Below 5 minute wait, never accept premium
                                             sf.ConstantDist(45).generate,  # Max price paid at 50 minutes (5 + 45) with 0 premium wait time
                                             sf.ConstantDist(10).generate,  # Max price willing to pay is $10.00 per seat/unit
                                             sf.ConstantDist(0.8).generate) # Premium must reduce wait time by at least 80%
    df = cdmg.generate_decision_f()
    reg_wait_time = 55
    prem_wait_time = 0
    prem_price_quote = 15
    party_size = 2
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'premium')
    reg_wait_time = 70
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'premium')
    prem_price_quote = 25
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'balk') # Max premium accept price is $20 and regular wait time is 70 minutes, so balk
    reg_wait_time = 30
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'regular') # Max premium accept price still $20, but regular wait time is 30 minutes (acceptable)
    reg_wait_time = 27.5   # Exactly half-way to max price threshold, so max accept price should be 10
    prem_price_quote = 9.9 # This is below 10, so premium offer should be accepted
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'premium')
    prem_price_quote = 10  # This is exactly 10, so premium offer should still be accepted
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'premium')
    prem_price_quote = 10.1 # This is above 10, so premium offer should be rejected for regular
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'regular')
    reg_wait_time = 50 # This results in max premium accept price of 20 if premium wait were 0
    prem_wait_time = 5 # This reduces the accept price by 5/50 = 10% -> max aacept price should be 18
    prem_price_quote = 15
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'premium')
    prem_price_quote = 18.1 # Offer should now be rejected for regular
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'regular')
    prem_price_quote = 0.01 # Offer should now be rejected for regular
    prem_wait_time = 15 # The current premium/regular waiting time ratio is 15/50 = 0.3, so reduction by 70% < required 80%. Therefore, premium should be rejected for regular.
    assert(df(reg_wait_time, prem_wait_time, prem_price_quote, party_size) == 'regular')
    print('Customer Decision Model Generator Test Passed')   


# ----------- Test the Simulation Time Rule Agent -------
def test_sim_time_rule_agent():
    class MockSimEnv:
        def __init__(self, now=0):
            self.now = now
    env = MockSimEnv()
    max_request_size = 6
    rules = [(10, (0, 0)), (20, (5, 10)), (30, (50, 5))]
    tra = sf.SimTimeRuleControlBoard(env, max_request_size, rules)
    request_size = 1
    assert(tra.premium_price(request_size) == float('infinity') and tra.premium_capacity_level() == 0)
    env.now = 5
    assert(tra.premium_price(request_size) == float('infinity') and tra.premium_capacity_level() == 0)
    env.now = 10
    assert(tra.premium_price(request_size) == float('infinity') and tra.premium_capacity_level() == 0)
    env.now = 12
    assert(tra.premium_price(request_size) == 5 and tra.premium_capacity_level() == 10)
    env.now = 20
    assert(tra.premium_price(request_size) == 5 and tra.premium_capacity_level() == 10)
    env.now = 21
    assert(tra.premium_price(request_size) == 50 and tra.premium_capacity_level() == 5)
    request_size = 3
    env.now = 30
    assert(tra.premium_price(request_size) == 150 and tra.premium_capacity_level() == 5)
    request_size = 4
    assert(tra.premium_price(request_size) == 200 and tra.premium_capacity_level() == 5)
    env.now = 31
    assert(tra.premium_price(request_size) == float('infinity') and tra.premium_capacity_level() == 0)
    print('Simulation Time Rule Agent Test Passed')


# ----------- Main Test Calls ---------------------------
test_constant_dist()
test_discrete_uniform_dist()
test_continuous_uniform_dist()
test_exp_interarrival_dist()
test_uncensored_normal_dist()
test_censored_normal_dist()
test_histogram_dist()
test_sim_time_dependent_dist()
test_customer_decision_model_generator()
test_sim_time_rule_agent()