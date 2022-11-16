import os.path

import numpy as np
from matplotlib import pyplot as plt
from numpy import floor

from martingale_optimal_transport.discrete import MartingaleOptimalTransport, RelaxedMartingaleOptimalTransport
from robust_pricing.generators.binomial_generator import BinomialGenerator

from gurobipy import GRB

from robust_pricing.generators.black_scholes import BlackScholesDiscretization


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
    
    n, m = 10, 2
    
    fig_path = os.path.join('data', 'Figures', f'experiment_BS(n={n}m={m},t={t})')
    
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
        sense=GRB.MINIMIZE,
        epsilon=epsilon
    )
    
    mot.draw_lp_matrix(save_name=os.path.join(fig_path, 'matrix.png'), plotly=False)
    mot.draw_lp_matrix(save_name=os.path.join(fig_path, 'matrix.png'), plotly=True)
    plt.show()
    
    coupling = mot.solve()
    
    os.makedirs(fig_path, exist_ok=True)
    
    for i in range(instance.time_horizon):
        coupling.draw_slice(i)
        plt.savefig(os.path.join(fig_path, f'S{i + 1}-{i + 2}.png'), dpi=360)
    
    # plt.show()
    

if __name__ == '__main__':
    rmot_instance()
