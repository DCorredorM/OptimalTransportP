from abc import abstractmethod, ABC
from typing import List

from measure_spaces import DiscreteMeasure


class MarginalsGenerator(ABC):
    def __init__(self, time_horizon):
        self.time_horizon = time_horizon
        self._marginals: List[DiscreteMeasure] = [
            self._time_t_marginal(t) for t in range(self.time_horizon + 1)
        ]
    
    @abstractmethod
    def _time_t_marginal(self, t) -> DiscreteMeasure:
        ...
    
    @property
    def marginals(self):
        return self._marginals
