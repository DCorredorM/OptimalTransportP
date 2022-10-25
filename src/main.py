from matplotlib import pyplot as plt

from martingale_optimal_transport.discrete import MartingaleOptimalTransport, RelaxedMartingaleOptimalTransport
from robust_pricing.generators.binomial_generator import BinomialGenerator

from gurobipy import GRB

from robust_pricing.generators.black_scholes import BlackScholesDiscretization


def mot_instance():
    bt = BinomialGenerator(
        100, 1.1, 6, probability=0.5
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
    mot.solve()


def rmot_instance():
    instance = BlackScholesDiscretization(
        time_horizon=3,
        discretization_kwargs=dict(n=3, m=3)
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
        epsilon=0.15
    )
    coupling = mot.solve()
    
    coupling.draw_slice(0)
    plt.show()
    coupling.draw_slice(1)
    plt.show()
    

if __name__ == '__main__':
    rmot_instance()
