from enum import Enum

import numpy as np

import scipy.stats as sts
import scipy.optimize as opt


class DiscretizationType(Enum):
    DETERMINISTIC = "DETERMINISTIC"
    RANDOM = "RANDOM"
    

class Discretization:
    @staticmethod
    def deterministic_discretization(scipy_continuous: sts.rv_continuous, n: int, m: int, **kwargs):
        from measure_spaces import DiscreteMeasure
        support = list(map(lambda i: i / n, range(n * m)))
        mass = [
            Discretization._get_x_i_k(scipy_continuous, n, i) / n
            for i in range(1, n * m)
        ]
        mass = [1 - sum(mass)] + mass
        
        return DiscreteMeasure(support=support, mass=mass, **kwargs)

    @staticmethod
    def _get_x_i_k(measure: sts.rv_continuous, n: int, i: int) -> float:
        bounds = opt.Bounds(lb=i / n, ub=(i + 1) / n, keep_feasible=True)
        result = opt.minimize(fun=measure.pdf, x0=np.array([bounds.lb]), bounds=bounds)
        return result.fun[0]

    @staticmethod
    def random_discretization(scipy_continuous: sts.rv_continuous, bins=None, **kwargs):
        from measure_spaces import DiscreteMeasure
        # TODO: Fix this method
        n = 1000
        support, mass = np.histogram(scipy_continuous.rvs(n), bins=bins)
        support = (support[:-1] + support[1:]) / 2
        mass = mass / n
        return DiscreteMeasure(support, mass, **kwargs)
        
        
        
        
        

    