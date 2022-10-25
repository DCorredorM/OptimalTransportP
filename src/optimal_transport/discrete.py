from itertools import product
from typing import Optional

from gurobipy import GRB

from optimal_transport._base import *
import gurobipy as gp
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
import pandas as pd
import numpy as np

from optimal_transport._base import OptimalTransportMulti as _OPM

from math import prod

from optimal_transport.solution_object import OptimalTransportSolution

COUPLING_NAME = 'coupling'


class OptimalTransport(_OPM):
    INFEASIBILITY_MESSAGE = "The given OT instance is infeasible."
    
    def __init__(
            self,
            mu: List[DiscreteMeasure],
            cost_function: Callable,
            sense=None
    ):
        super().__init__(mu, cost_function)
        self._solution_object = None
        self.sense = sense
        self.mu: List[DiscreteMeasure]
        
        self.model = gp.Model()
        self.model.setParam('OutputFlag', 1)
        
        # full of expres variables
        self._coupling: Optional[np.ndarray] = None
        
        # the actual coupling, i.e., a solution
        self.coupling: Optional[np.ndarray] = None
        
        self.dimensions = tuple([len(m.support) for m in self.mu])
        
        self.logger.info(f"The model will have {prod(self.dimensions)} variables. ")
        
        self._build_model()
        self.model.update()
    
    # Variables
    def _create_coupling_variable(self):
        # add variables
        prob = self.model.addMVar(shape=self.dimensions, lb=0, ub=1, vtype=GRB.CONTINUOUS, name=COUPLING_NAME)
        self._coupling = np.array(prob.tolist())
    
    # Constraints
    def _create_marginals_constraint(self):
        
        base_idx = tuple(list(range(len(self.mu))))
        for t, mu in enumerate(self.mu):
            idx = list(base_idx)
            idx.remove(t)
            for i, (lhs, rhs) in enumerate(zip(self._coupling.sum(axis=tuple(idx)), mu.mass)):
                self.model.addConstr(lhs == rhs, name=f"marginal{t}{i}")
    
    # Objective value
    def _create_objective_value(self):
        index_set = product(*[range(i) for i in self.dimensions])
        self.model.setObjective(
            gp.quicksum(
                self._coupling[i] * self.cost_function([self.mu[t].support[i[t]] for t in range(len(self.mu))])
                for i in index_set
            ),
            sense=self.sense
        )
    
    def _build_model(self):
        
        # create_variables
        self._create_coupling_variable()
        
        # add marginals constraints
        self._create_marginals_constraint()
        
        # add objective function
        self._create_objective_value()
        
        self.model.update()
    
    def solve(self):
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            self._extract_solution()
        else:
            self.logger.warning(self.__class__.INFEASIBILITY_MESSAGE)
        
        return self.solution_object
    
    def _extract_solution(self):
        @np.vectorize
        def _extract(var):
            return var.x
        
        self.coupling = _extract(self._coupling)
        return self.coupling
    
    @property
    def solution_object(self):
        if self._solution_object is None:
            if self.coupling is None:
                raise ValueError("Problem has not been yet solved. Solve the problem first and the you can query the "
                                 "solution object.")
            else:
                self._solution_object = OptimalTransportSolution(self.coupling, self)
        return self._solution_object


class DiscreteOT(OptimalTransport):
    def __init__(self, alpha: DiscreteMeasure, beta: DiscreteMeasure, cost_function: Callable):
        super().__init__([alpha, beta], cost_function)
        self.alpha = alpha
        self.beta = beta
    
    def plot(self, grid=True, hide_ticks=False, kind='hist'):
        self.solution_object.draw_slice(0, grid=grid, hide_ticks=hide_ticks, kind=kind)
    
    def plot_cost_function(self, resolution=200):
        xlist = np.linspace(-max(self.alpha.support), max(self.alpha.support), resolution)
        ylist = np.linspace(-max(self.beta.support), max(self.beta.support), resolution)
        
        X, Y = np.meshgrid(xlist, ylist)
        M = self.create_nodes(xlist, ylist)
        Z = np.apply_along_axis(self.cost_function, 2, M)
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.contourf(X, Y, Z)
    
    def create_nodes(self, X=None, Y=None):
        X = self.alpha.support if X is None else X
        Y = self.beta.support if Y is None else Y
        M = np.array([[[x, y] for y in Y] for x in X])
        
        return M


if __name__ == '__main__':
    import scipy.stats as sts
    
    n = 1000
    bins = 30
    
    x = np.concatenate([
        sts.gamma(1).rvs(int(n / 2)),
        sts.gamma(4, loc=5).rvs(int(n / 2))])
    y = sts.gamma(10).rvs(n)
    
    mass_x, support_x = np.histogram(x, bins=bins)
    support_x = (support_x[:-1] + support_x[1:]) / 2
    
    alpha = DiscreteMeasure(support=support_x, mass=mass_x / n)
    
    mass_y, support_y = np.histogram(y, bins=bins)
    support_y = (support_y[:-1] + support_y[1:]) / 2
    beta = DiscreteMeasure(support=support_y, mass=mass_y / n)
    
    
    def build_f(M):
        def f(x_bar):
            if isinstance(x_bar, list):
                x_bar = np.array(x_bar)
            return np.dot(np.dot(M, x_bar), x_bar.T)
        
        return f
    
    
    M = np.array([[1, 0],
                  [0, 1]])
    
    f = build_f(M)
    
    ot = DiscreteOT(alpha, beta, cost_function=f)
    ot.solve()
    # ot.plot()
    ot.plot_cost_function()
    plt.show()
