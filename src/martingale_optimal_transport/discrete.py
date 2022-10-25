from itertools import product
from typing import List, Callable, Optional

from measure_spaces import DiscreteMeasure, DiscreteMeasureSpace
from optimal_transport import OptimalTransport

import gurobipy as gp
from gurobipy import GRB


class MartingaleOptimalTransport(OptimalTransport):
    INFEASIBILITY_MESSAGE = "There is no Martingale measure that satisfy the given marginals."
    
    def _create_martingale_constraint(self):
        for t in range(len(self.mu) - 1):
            for i_0 in product(*[range(i) for i in self.dimensions[:t+1]]):
                self.model.addConstr(
                    gp.quicksum(
                        self._coupling[i_0 + i_1] * (self.mu[t+1].support[i_1[0]] - self.mu[t].support[i_0[-1]])
                        for i_1 in product(*[range(i) for i in self.dimensions[t+1:]])
                    ) == 0,
                    name=f"martingale{t}{i_0}"
                )
    
    def _build_model(self):
    
        # create_variables
        self._create_coupling_variable()
    
        # add marginals constraints
        self._create_marginals_constraint()

        # add martingale constraints
        self._create_martingale_constraint()
    
        # add objective function
        self._create_objective_value()
    
        self.model.update()


class RelaxedMartingaleOptimalTransport(OptimalTransport):
    
    def __init__(self, mu: List[DiscreteMeasure], cost_function: Callable, sense=None, epsilon=1e-4):
        self.epsilon = epsilon
    
        # Auxiliary variable to account for the sum within constraint 2.1.c
        self._x_i = {}
    
        # Auxiliary variable to account for the absolute value of x_i
        self._y_i = {}
        
        super().__init__(mu, cost_function, sense)
    
    def _create_auxiliary_variables(self):
        for t in range(len(self.mu) - 1):
            for i_0 in product(*[range(i) for i in self.dimensions[:t + 1]]):
                self._x_i[i_0] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"sum{i_0}")
                self._y_i[i_0] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"abs{i_0}")
                
                self.model.addConstr(
                    self._x_i[i_0] == gp.quicksum(
                        self._coupling[i_0 + i_1] * (self.mu[t + 1].support[i_1[0]] - self.mu[t].support[i_0[-1]])
                        for i_1 in product(*[range(i) for i in self.dimensions[t + 1:]])
                    ),
                    name=f"sumConstr{i_0}"
                )
                self.model.addGenConstrAbs(self._y_i[i_0], self._x_i[i_0], name=f'absConstr{i_0}')
                
    def _create_martingale_constraint(self):
        for t in range(len(self.mu) - 1):
            self.model.addConstr(
                gp.quicksum(self._y_i[i_0] for i_0 in product(*[range(i) for i in self.dimensions[:t + 1]]))
                <= self.epsilon
            )
    
    def _build_model(self):
        
        # create_variables
        self._create_coupling_variable()
        
        # create auxiliary variables for abs
        self._create_auxiliary_variables()
        
        # add marginals constraints
        self._create_marginals_constraint()
        
        # add martingale constraints
        self._create_martingale_constraint()
        
        # add objective function
        self._create_objective_value()
        
        self.model.update()