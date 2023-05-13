from typing import Dict

import numpy as np
from scipy import stats as sts
from measure_spaces import DiscreteMeasure, DiscretizationType
from robust_pricing.marginals_creators._base import MarginalsGenerator


class BlackScholesMeasure(sts.rv_continuous):
    def __init__(self, k, N):
        self.k = k
        self.N = N
        
        super().__init__()
    
    def _pdf(self, x, *args):
        if x > 0:
            return (np.exp(- (np.log(x) + 2 ** (self.k - 4))**2 / (2 ** (self.k - 2)))) \
                   / (x * np.sqrt(2 ** (self.k - 2) * np.pi))
        else:
            return 0


class BlackScholesDiscretization(MarginalsGenerator):
    def __init__(
            self,
            time_horizon,
            discretization: DiscretizationType = DiscretizationType.DETERMINISTIC,
            discretization_kwargs: Dict = None
    ):
        self.discretization = discretization
        if discretization_kwargs is None:
            self.discretization_kwargs = dict(n=100, m=100)
        else:
            self.discretization_kwargs = discretization_kwargs
        
        super().__init__(time_horizon - 1)
        
    def _time_t_marginal(self, t):
        rv_cont = BlackScholesMeasure(k=t + 1, N=self.time_horizon + 1)
        return DiscreteMeasure.from_scipy_continuous(
            scipy_continuous=rv_cont,
            discretization=self.discretization,
            discretization_kwargs=self.discretization_kwargs
        )