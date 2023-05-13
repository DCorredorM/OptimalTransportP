import os.path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy import floor
from scipy.stats import rv_continuous
from scipy.integrate import quad
import seaborn as sns

from martingale_optimal_transport.discrete import MartingaleOptimalTransport, RelaxedMartingaleOptimalTransport
from measure_spaces import Measure, DiscreteMeasure, DiscretizationType
from optimal_transport import OptimalTransport, COUPLING_NAME, OptimalTransportSolution
from robust_pricing.marginals_creators.binomial_generator import BinomialGenerator

import gurobipy as gp
from gurobipy import GRB

from robust_pricing.marginals_creators.black_scholes import BlackScholesDiscretization
from scipy.stats.sampling import TransformedDensityRejection


def mot_instance():
    t = 6
    bt = BinomialGenerator(
        spot=100, up=1.1, time_horizon=t, probability=0.5
    )
    
    def cost_function(history):
        return max(sum(history) / len(history) - 100, 0)
        # return max(max(history) - 100, 0)
        # return max(history[-1] - 100, 0)
    
    mot = MartingaleOptimalTransport(
        mu=[bt.marginals[1], bt.marginals[-1]],
        cost_function=cost_function,
        sense=GRB.MINIMIZE
    )

    fig_path = os.path.join('data', 'Figures', f'experiment_BG(t={t})')

    mot.draw_lp_matrix(save_name=os.path.join(fig_path, 'matrix.png'))
    plt.show()
    
    mot.solve()


def rmot_instance():
    
    t = 3
    n = 50
    theta = 1.8
    L = 12
    M = np.exp(theta * (theta - 1))
    m = floor((n * (theta - 1) * M / L) ** (1 / (theta + 1)))
    epsilon = 3 * (1 / n + 5 * M / (m ** (theta - 1)) + 2 * m ** 2 * L / n)
    # epsilon = 0.14
    
    # n, m = 2, 3
    
    fig_path = os.path.join('data', 'Figures', f'experiment_BS(n={n},m={m},t={t}),epsilon={epsilon}')
    
    instance = BlackScholesDiscretization(
        time_horizon=t,
        discretization_kwargs=dict(n=n, m=m)
    )
    
    def look_back(history):
        return max(history) - history[-1]

    lambda_ = 2
    
    def asian(history):
        return max(sum(history) / len(history) - lambda_ * history[-1], 0)
    
    mot = RelaxedMartingaleOptimalTransport(
        mu=instance.marginals,
        cost_function=look_back,
        sense=GRB.MAXIMIZE,
        epsilon=epsilon
    )
    
    # mot.draw_lp_matrix(save_name=os.path.join(fig_path, 'matrix.png'), plotly=False)
    # mot.draw_lp_matrix(save_name=os.path.join(fig_path, 'matrix.png'), plotly=True)
    # plt.show()

    coupling = mot.solve()
    
    os.makedirs(fig_path, exist_ok=True)
    
    for i in range(instance.time_horizon):
        coupling.draw_slice(i)
        plt.savefig(os.path.join(fig_path, f'S{i + 1}-{i + 2}.png'), dpi=360)
    
    plt.show()
    

def gen_example37_alpha_beta(n=100, ):
    class MuGen(rv_continuous):
        "Mu distribution"
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.c, _ = quad(MuGen._un_normalized_pdf, 0, 1)
        
        @staticmethod
        def _un_normalized_pdf(x):
            return int(0 <= x <= 1) * x ** (3 / 2) * np.exp(-x)
        
        def _pdf(self, x, **kwargs):
            return MuGen._un_normalized_pdf(x) / self.c
        
        # def dpdf(self, x):
        #     return 2 / 3 * x ** (1 / 2) * np.exp(-x) - x ** (3 / 2) * np.exp(-x)
        #
        # def _rvs(self, *args, size=None, random_state=None):
        #     urng = np.random.default_rng()
        #     rng = TransformedDensityRejection(self, random_state=urng)
        #     return rng.rvs(size=size)
    
    mu = MuGen()
    
    alpha = DiscreteMeasure.from_scipy_continuous(
        scipy_continuous=MuGen(),
        discretization=DiscretizationType.DETERMINISTIC,
        discretization_kwargs=dict(n=n, m=1)
    )
    
    print(len(alpha.mass), len(alpha.support))
    plt.bar(alpha.support, alpha.mass, width=0.001)
    plt.show()
    
    class NuGen(rv_continuous):
        "Mu distribution"
        
        def __init__(self, rho, **kwargs):
            super().__init__(**kwargs)
            self.rho: Callable = rho
        
        def _pdf(self, x, **kwargs):
            return self.rho(x / 2) / 6 + 4 * self.rho(2 * x) / 3
    
    nu = NuGen(rho=mu.pdf)
    
    beta = DiscreteMeasure.from_scipy_continuous(
        scipy_continuous=nu,
        discretization=DiscretizationType.DETERMINISTIC,
        discretization_kwargs=dict(n=n, m=2)
    )

    print(len(beta.mass), len(beta.support))
    plt.bar(beta.support, beta.mass, width=0.001)
    plt.show()
    
    return alpha, beta, mu, nu


def relaxed_mot_instance_1():
    """
    In this function we implement the instance presented in Example 3.7 of [1]_.
    """
    # First we define the measures:
    n = 100
    L = 7
    epsilon = (3 * L + 2) / n
    
    alpha, beta, _, _ = gen_example37_alpha_beta(n=n)
    
    def h(history):
        return np.exp(history[0] - history[-1])
     
    mot = RelaxedMartingaleOptimalTransport(
        mu=[alpha, beta],
        cost_function=h,
        sense=GRB.MAXIMIZE,
        epsilon=epsilon
    )

    coupling = mot.solve()

    # os.makedirs(fig_path, exist_ok=True)

    coupling.draw_slice(0)
    # plt.savefig(os.path.join(fig_path, f'S{i + 1}-{i + 2}.png'), dpi=360)

    plt.show()


def example_37_plain():
    n = 300
    L = 7
    epsilon = (3 * L + 2) / n

    alpha, beta, mu, nu = gen_example37_alpha_beta(n=n)
    
    alpha_i = [mu.pdf((i / n + ((i + 1) / n)) / 2) / n for i in range(1, n)]
    alpha_i = [abs(1 - sum(alpha_i))] + alpha_i
    alpha_i = [a / sum(alpha_i) for a in alpha_i]

    beta_i = [nu.pdf((i / n + ((i + 1) / n)) / 2) / n for i in range(1, 2 * n)]
    beta_i = [abs(1 - sum(beta_i))] + beta_i
    beta_i = [a / sum(beta_i) for a in beta_i]
    
    # Create the model
    model = gp.Model()
    model.setParam('OutputFlag', 1)
    
    # Add the p_ij variable:
    prob = model.addMVar(shape=(n, 2 * n), lb=0, ub=1, vtype=GRB.CONTINUOUS, name=COUPLING_NAME)
    prob = np.array(prob.tolist())
    
    # Add the OT constraints
    
    for i in range(n):
        model.addConstr(gp.quicksum(prob[i, j] for j in range(2 * n)) == alpha_i[i])
    
    for j in range(2 * n):
        model.addConstr(gp.quicksum(prob[i, j] for i in range(n)) == beta_i[j])
    
    
    # Add the relazed mot variables
    
    sum_i = dict()
    abs_i = dict()

    for i in range(n):
        sum_i[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"sum_{i}")
        abs_i[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"abs_{i}")

        model.addConstr(
            sum_i[i] == gp.quicksum(
                prob[(i, j)] * j / n for j in range(2 * n)
            ) - alpha_i[i] * i / n,
            name=f"sumConstr_{i}"
        )

        model.addGenConstrAbs(abs_i[i], sum_i[i], name=f'absConstr_{i}')

    model.addConstr(
        gp.quicksum(abs_i[i] for i in range(n)) <= epsilon,
        name=f"martingale_({0},)"
    )

    model.setObjective(
        gp.quicksum(
            prob[i, j] * np.exp((i - j) / n) for i in range(n) for j in range(2 * n)
        ),
        sense=GRB.MAXIMIZE,
    )

    model.update()

    # ASolve instance
    model.optimize()

    @np.vectorize
    def _extract(var):
        return var.x

    coupling = _extract(prob)

    sol = OptimalTransportSolution(
        coupling=coupling,
        instance=OptimalTransport(
            mu=[alpha, beta],
            cost_function=lambda x: x,
            build=False
        )
    )

    sol.draw_slice(0)
    
    plt.show()


if __name__ == '__main__':
    # example_37_plain()
    # relaxed_mot_instance_1()
    rmot_instance()
