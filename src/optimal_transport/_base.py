from measure_spaces import *
from typing import Callable, List
from abc import ABC
import logging


class OptimalTransport(ABC):
    def __init__(self,
                 alpha: Measure,
                 beta: Measure,
                 cost_function: Callable):
        self.alpha = alpha
        self.beta = beta
        self.cost_function = cost_function
        self.logger = logging.getLogger(name=self.__class__.__name__)

    @abstractmethod
    def solve(self):
        ...


class OptimalTransportMulti(ABC):
    def __init__(self,
                 mu: List[Measure],
                 cost_function: Callable):
        self.mu = mu
        self.cost_function = cost_function
        self.logger = logging.getLogger(name=self.__class__.__name__)

    @abstractmethod
    def solve(self):
        ...
    
    
