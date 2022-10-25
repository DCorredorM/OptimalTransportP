from scipy import stats as sts
from measure_spaces import DiscreteMeasure
from robust_pricing.generators._base import MarginalsGenerator


class BinomialGenerator(MarginalsGenerator):
    def __init__(self, spot, up, time_horizon, probability=None):
        self.spot = spot
        self.up = up
        if probability is None:
            self.probability = (spot - spot / up) / (spot * up - spot / up)
        else:
            self.probability = probability
        
        super().__init__(time_horizon)

    def _time_t_marginal(self, t):
        s, u, d = self.spot, self.up, 1 / self.up
        support = [s * u ** ups * d ** (t - ups) for ups in range(t + 1)]
        return DiscreteMeasure(support=support, density=sts.binom(n=t, p=self.probability))