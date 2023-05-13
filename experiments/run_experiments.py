import json

import os

import numpy as np
from copy import deepcopy
from gurobipy import GRB

from typing import Tuple

from martingale_optimal_transport.discrete import MartingaleOptimalTransport
from robust_pricing.deePricing import HedgingStrategy, Call, Put, OneMaturityOptionPortfolio, DiagonalOptionPortfolio
import torch

from robust_pricing.path_generators import BinomialTree, GaussianMartingale, UniformMartingale, Uniform, Gaussian
from robust_pricing.marginals_creators.binomial_generator import BinomialGenerator


import boto3

WRITE_LOCAL = True
BUCKET_NAME = 'mot-experiments'
NUMBER_OF_GAMMAS = 30
EXPLORATION_POWER = 3


def option_generate(pricing_model, path_length, ):
    strike_prices = list(range(70, 130, 1))
    pricing_data = pricing_model(10000)
    
    option_portfolios = []
    
    for t in range(path_length):
        calls = []
        puts = []
        for k in strike_prices:
            c = Call(strike=k, price=0)
            c.price = float(torch.mean(c(pricing_data[:, t])))
            p = Put(strike=k, price=0)
            p.price = float(torch.mean(p(pricing_data[:, t])))
            calls.append(c)
            puts.append(p)
        
        option_portfolios.append(
            OneMaturityOptionPortfolio(calls=calls, puts=puts)
        )
    return DiagonalOptionPortfolio(option_portfolios)


def build_base_binomial_model(path_length, granularity=100, volatility=0.15):
    
    # Full model
    up_factor = np.exp(volatility * np.sqrt(1 / granularity))
    down_factor = 1 / up_factor
    observed_times = [i * (granularity // path_length) for i in range(1, path_length + 1)]
    model = BinomialTree(
        path_length=path_length,
        mean=100,
        up_factor=up_factor,
        down_factor=down_factor,
        granularity=granularity,
        observed_times=observed_times,
    )
    
    # marginals model
    mu = BinomialGenerator(100, up=up_factor, time_horizon=granularity)
    marginals = [mu.marginals[i] for i in observed_times]
    
    return model, marginals


def get_data_s3_bucket():
    bucket = boto3.client('s3')
    return bucket


def write_data(base, key, data, write_local=True):
    try:
        bucket = get_data_s3_bucket()
        bucket.put_object(
            Body=json.dumps(data),
            Bucket=base,
            Key=key
        )
    except Exception as e:
        print(e.with_traceback(e.__traceback__))
    
    if write_local:
        os.makedirs(os.path.join(base, *key.split('/')[:-1]), exist_ok=True)
        with open(os.path.join(base, key), 'w') as f:
            f.write(json.dumps(data))


def run_experiment_binomial(
        base_name,
        option_portfolio,
        marginals,
        sampling_model,
        target_function,
        test_data=None,
        solve_discrete_version=False,
        check_if_exists=True,
        **kwargs
):
    # return {"sup_replication_value": np.random.random()*100, "sub_replication_value": np.random.random()*100}
    gamma = kwargs.get("gamma", 300)
    experiment_data = {
        "name": os.path.join(
            f'experiment_name={base_name}',
            f'num_marginals={len(marginals)}',
            f'sampling_model={sampling_model.__class__.__name__}',
            f'gamma={gamma}',
            f'id={hash(str(sorted(tuple(kwargs.items()))))}'
        )
    }
    
    path = os.path.join(BUCKET_NAME, experiment_data["name"])
    if os.path.exists(path) and check_if_exists:
        with open(os.path.join(path, 'experiment_summary.json'), 'r') as f:
            return json.loads(f.read())
    
    zeros = kwargs.get("zeros", True)
    
    number_of_observations = kwargs.get("number_of_observations", 1000)
    no_trading_strategy = kwargs.get("no_trading_strategy", False)
    
    experiment_data.update(
        dict(
            zeros=zeros,
            gamma=gamma,
            number_of_observations=number_of_observations,
            time_horizon=len(marginals),
            sampling_model=sampling_model.__class__.__name__,
            no_trading_strategy=no_trading_strategy,
            binomial_model_granularity=kwargs.get("binomial_model_granularity"),
            binomial_model_up_factor=kwargs.get("binomial_model_up_factor"),
            binomial_model_observed_times=kwargs.get("binomial_model_observed_times"),
        )
    )

    # ############################################################
    # build dual-mot
    # ############################################################

    f = target_function
    option_portfolio.reset_weights(zeros=zeros)

    trading_kwargs = kwargs.get("trading_kwargs", dict(num_layers=5, width_layers=100))
    
    # define sup/sub headging strategies
    h_sup = HedgingStrategy(
        option_portfolio=deepcopy(option_portfolio),
        target_function=f,
        super_hedge=True,
        no_trading_strategy=no_trading_strategy,
        trading_kwargs=trading_kwargs
    )
    h_sub = HedgingStrategy(
        option_portfolio=deepcopy(option_portfolio),
        target_function=f,
        super_hedge=False,
        no_trading_strategy=no_trading_strategy,
        trading_kwargs=trading_kwargs
    )

    # Set the given gamma
    h_sup.penalization_function.gamma = gamma
    h_sub.penalization_function.gamma = gamma

    # ############################################################
    # build lp
    # ############################################################
    if solve_discrete_version:
        mot_sup = MartingaleOptimalTransport(
            mu=marginals,
            cost_function=target_function,
            sense=GRB.MAXIMIZE
        )

        mot_inf = MartingaleOptimalTransport(
            mu=marginals,
            cost_function=target_function,
            sense=GRB.MINIMIZE
        )
        mot_sup.solve()
        mot_inf.solve()
    # ############################################################
    # solve models
    # ############################################################
    option_portfolio.reset_weights(zeros=zeros)
    
    # optimize for the given generator model
    h_sup.optimize(
        generator=sampling_model,
        number_of_observations=number_of_observations,
    )

    h_sub.optimize(
        generator=sampling_model,
        number_of_observations=number_of_observations,
    )

    # ############################################################
    # Record data
    # ############################################################
    
    # Replication errors
    if test_data is not None:
        experiment_data["evaluation_sup_replication_error"] = float(h_sup.replication_error(test_data).detach())
        experiment_data["evaluation_sub_replication_error"] = float(h_sub.replication_error(test_data).detach())
        
    experiment_data["in_sample_sup_replication_error"] = float(
        h_sup.training_status['replication_error'][-1]
    )
    experiment_data["in_sample_sup_replication_error"] = float(
        h_sub.training_status['replication_error'][-1]
    )
    
    # Upper and lowe bounds
    experiment_data["sup_replication_value"] = float(h_sup.value)
    experiment_data["sub_replication_value"] = float(h_sub.value)

    if solve_discrete_version:
        experiment_data["sup_replication_discrete_value"] = mot_sup.solution_object.optimal_value
        experiment_data["sub_replication_discrete_value"] = mot_inf.solution_object.optimal_value
        
        for attr in mot_sup.solution_stats.keys():
            experiment_data[f"sup_replication_discrete_{attr}"] = mot_sup.solution_stats.get(attr)
            experiment_data[f"sub_replication_discrete_{attr}"] = mot_inf.solution_stats.get(attr)

    # Execution times & other stats

    for attr, value in h_sup.training_status.items():
        if not isinstance(value, list):
            experiment_data[f"sup_replication_{attr}"] = h_sup.training_status.get(attr)
            experiment_data[f"sub_replication_{attr}"] = h_sub.training_status.get(attr)
        
    # experiment_data["sup_replication_execution_time"] = h_sup.training_status["total_execution_time"]
    # experiment_data["sub_replication_execution_time"] = h_sub.training_status["total_execution_time"]
    #
    # experiment_data["sup_replication_total_num_episodes"] = h_sup.training_status["total_num_episodes"]
    # experiment_data["sub_replication_total_num_episodes"] = h_sub.training_status["total_num_episodes"]
    #
    # experiment_data["sup_replication_best_bound"] = h_sup.training_status["best_bound"]
    # experiment_data["sub_replication_best_bound"] = h_sub.training_status["best_bound"]
    

    # ############################################################
    # Write data
    # ############################################################
    
    # write experiment summary
    write_data(
        base=BUCKET_NAME,
        key=os.path.join(experiment_data["name"], 'experiment_summary.json'),
        data=experiment_data,
        write_local=WRITE_LOCAL
    )

    write_data(
        base=BUCKET_NAME,
        key=os.path.join(experiment_data["name"], 'sup_training_data.json'),
        data=h_sup.training_status,
        write_local=WRITE_LOCAL
    )

    write_data(
        base=BUCKET_NAME,
        key=os.path.join(experiment_data["name"], 'sub_training_data.json'),
        data=h_sub.training_status,
        write_local=WRITE_LOCAL
    )
    
    return experiment_data


def get_next_gamma(**kwargs):
    tested_gammas = kwargs.get("tested_gammas")
    upper_bounds = kwargs.get("upper_bounds")
    lower_bounds = kwargs.get("lower_bounds")
    
    if len(tested_gammas) >= NUMBER_OF_GAMMAS:
        return False
    elif len(tested_gammas) <= EXPLORATION_POWER:
        return eval(f'1e{len(tested_gammas)}')
    else:
        all_zipped = sorted(zip(tested_gammas, upper_bounds, lower_bounds))
        
        def slope_score_function(x):
            (gamma_, u_, l_), (gamma, u, l) = x
            u_slope = abs(u_ - u)
            l_slope = abs(l_ - l)
            return (u_slope + l_slope) / 2
            
        scores = list(map(lambda x: slope_score_function(x), zip(all_zipped[1:], all_zipped[:-1])))
        i_star = max(range(len(scores)), key=lambda x: scores[x])
        star_gamma = all_zipped[i_star][0]
        star_gamma_ = all_zipped[i_star + 1][0]
        next_gamma = (star_gamma + star_gamma_) // 2
        times_tried = 0
        while next_gamma in tested_gammas:
            random_index = np.random.randint(len(tested_gammas))
            next_gamma = (next_gamma + tested_gammas[random_index]) // 2
            times_tried += 1
            if times_tried > 100:
                return (next_gamma + max(tested_gammas)) // 3
        
        return next_gamma
        
        
def loop_through_gammas(
        base_name,
        binomial_model,
        path_length,
        marginals,
        sampling_model,
        target_function,
        num_test_data,
        solve_discrete_version,
        number_of_observations,
        **kwargs
):
    state = dict(
        tested_gammas=[],
        upper_bounds=[],
        lower_bounds=[],
    )
    
    no_trading_strategy = kwargs.pop("no_trading_strategy", False)
    
    while gamma := get_next_gamma(**state):
        experiments_data = run_experiment_binomial(
            base_name=base_name,
            option_portfolio=option_generate(pricing_model=binomial_model, path_length=path_length),
            marginals=marginals,
            sampling_model=sampling_model,
            target_function=target_function,
            test_data=binomial_model(num_test_data),
            solve_discrete_version=solve_discrete_version,
            gamma=gamma,
            number_of_observations=number_of_observations,
            no_trading_strategy=no_trading_strategy,
            binomial_model_granularity=binomial_model.granularity,
            binomial_model_up_factor=binomial_model.up_factor,
            binomial_model_observed_times=binomial_model.observed_times,
            **kwargs
        )
        
        state["tested_gammas"].append(gamma)
        state["upper_bounds"].append(experiments_data["sup_replication_value"])
        state["lower_bounds"].append(experiments_data["sub_replication_value"])
        solve_discrete_version = False


def run_experiments_0(path_length=2, solve_discrete_version=True, no_trading_strategy=True):
    """
    Experiments 0: Can the hammer kill the fly 1

    We try to find the price of vanilla path independent option.

    1. We will solve the LP version of it
    2. We will solve for different gammas the Eckstein approach.
    """
    
    # path_length = 2
    binomial_model, marginals = build_base_binomial_model(path_length=path_length, volatility=0.15)
    # sampling_model = GaussianMartingale(path_length=path_length, mean=100, variance=20 ** 2)
    trading_kwargs = dict(num_layers=5, width_layers=100)
    
    num_test_data = 1000
    number_of_observations = 1000
    
    def target_function(x):
        K = 100
        if isinstance(x, torch.Tensor):
            return torch.relu(x.select(1, -1) - K)
        else:
            return max(x[-1] - K, 0)
    
    sampling_models = [
        Uniform(path_length, mean=binomial_model.mean, variance=80 ** 2),
        Gaussian(path_length, mean=binomial_model.mean, variance=30 ** 2),
        GaussianMartingale(path_length, mean=binomial_model.mean, variance=20 ** 2),
        UniformMartingale(path_length, mean=binomial_model.mean, variance=50 ** 2),
        binomial_model
    ]
    
    for sampling_model in sampling_models:
        loop_through_gammas(
            base_name='PriceCall100',
            binomial_model=binomial_model,
            path_length=path_length,
            marginals=marginals,
            sampling_model=sampling_model,
            target_function=target_function,
            num_test_data=num_test_data,
            solve_discrete_version=solve_discrete_version,
            number_of_observations=number_of_observations,
            no_trading_strategy=no_trading_strategy,
            trading_kwargs=trading_kwargs
        )


def run_experiments_1(path_length=2, solve_discrete_version=True, no_trading_strategy=True):
    """
    Experiments 1: Floating strike Asian option.

    We try to find the price of vanilla path independent option.

    1. We will solve the LP version of it
    2. We will solve for different gammas the Eckstein approach.
    """
    
    # path_length = 2
    binomial_model, marginals = build_base_binomial_model(path_length=path_length, volatility=0.15)
    # sampling_model = GaussianMartingale(path_length=path_length, mean=100, variance=20 ** 2)
    trading_kwargs = dict(num_layers=5, width_layers=100)
    
    num_test_data = 1000
    number_of_observations = 1000
    
    def target_function(x):
        if isinstance(x, torch.Tensor):
            return torch.relu(x.select(1, -1) - torch.mean(x, 1))
        else:
            return max(x[-1] - sum(x) / len(x), 0)
    
    sampling_models = [
        Uniform(path_length, mean=binomial_model.mean, variance=80 ** 2),
        Gaussian(path_length, mean=binomial_model.mean, variance=30 ** 2),
        GaussianMartingale(path_length, mean=binomial_model.mean, variance=20 ** 2),
        UniformMartingale(path_length, mean=binomial_model.mean, variance=50 ** 2),
        binomial_model
    ]
    
    for sampling_model in sampling_models:
        loop_through_gammas(
            base_name='FloatingPointAsian',
            binomial_model=binomial_model,
            path_length=path_length,
            marginals=marginals,
            sampling_model=sampling_model,
            target_function=target_function,
            num_test_data=num_test_data,
            solve_discrete_version=solve_discrete_version,
            number_of_observations=number_of_observations,
            no_trading_strategy=no_trading_strategy,
            trading_kwargs=trading_kwargs
        )


if __name__ == '__main__':
    # run_experiments_0(path_length=2, no_trading_strategy=True)
    # run_experiments_0(path_length=3, no_trading_strategy=True)
    # run_experiments_0(path_length=5, solve_discrete_version=False, no_trading_strategy=True)
    #
    # run_experiments_1(path_length=2, no_trading_strategy=True)
    # run_experiments_1(path_length=3, no_trading_strategy=True)
    # run_experiments_1(path_length=5, solve_discrete_version=False, no_trading_strategy=True)
    
    run_experiments_0(path_length=2, no_trading_strategy=False)
    run_experiments_0(path_length=3, no_trading_strategy=False)
    run_experiments_0(path_length=5, solve_discrete_version=False, no_trading_strategy=False)

    run_experiments_1(path_length=2, no_trading_strategy=False)
    run_experiments_1(path_length=3, no_trading_strategy=False)
    run_experiments_1(path_length=5, solve_discrete_version=False, no_trading_strategy=False)
