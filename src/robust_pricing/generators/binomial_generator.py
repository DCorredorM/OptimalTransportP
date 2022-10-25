from scipy import stats as sts
from measure_spaces import DiscreteMeasure


class BinomialGenerator:
    def __init__(self, spot, up, time_horizon, probability=None):
        
        self.spot = spot
        self.up = up
        if probability is None:
            self.probability = (spot - spot / up) / (spot * up - spot / up)
        else:
            self.probability = probability
            
        self.time_horizon = time_horizon
        
        self._marginals = [self._time_t_marginal(t) for t in range(self.time_horizon+1)]

    def _time_t_marginal(self, t):
        s, u, d = self.spot, self.up, 1 / self.up
        support = [s * u ** ups * d ** (t - ups) for ups in range(t + 1)]
        return DiscreteMeasure(support=support, density=sts.binom(n=t, p=self.probability))
    
    @property
    def marginals(self):
        return self._marginals