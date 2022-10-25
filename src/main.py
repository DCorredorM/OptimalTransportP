from martingale_optimal_transport.discrete import MartingaleOptimalTransport
from robust_pricing.generators.binomial_generator import BinomialGenerator

from gurobipy import GRB


if __name__ == '__main__':
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
