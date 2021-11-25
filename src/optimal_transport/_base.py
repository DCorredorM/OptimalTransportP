from measure_spaces import *
from typing import Callable
from abc import ABC


class OptimalTransport(ABC):
	def __init__(self,
	             alpha: Measure,
	             beta: Measure,
	             cost_function: Callable):
		self.alpha = alpha
		self.beta = beta
		self.cost_function = cost_function

	@abstractmethod
	def solve(self):
		...
