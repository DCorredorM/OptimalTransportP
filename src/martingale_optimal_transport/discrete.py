from itertools import product
from typing import List, Callable, Optional

from measure_spaces import DiscreteMeasure, DiscreteMeasureSpace
from optimal_transport import OptimalTransport

import gurobipy as gp


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
