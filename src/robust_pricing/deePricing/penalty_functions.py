from typing import Callable

from enum import Enum

import torch


class PenaltyFunctionTypes(Enum):
    lp_norm = 0
    exponential = 1


class PenaltyFunction:
    GAMMA = 1
    DEFAULT_P = 2
    
    def __init__(
            self,
            initial_gamma: float = None,
            penalty_type: PenaltyFunctionTypes = PenaltyFunctionTypes.lp_norm,
            penalty_type_kwargs=None
    ):
        
        if penalty_type_kwargs is None:
            penalty_type_kwargs = {}

        from robust_pricing.deePricing import PenaltyFunction
        
        self.gamma = PenaltyFunction.GAMMA if initial_gamma is None else initial_gamma
        self.penalty_type = penalty_type
        
        if penalty_type.name == PenaltyFunctionTypes.lp_norm.name:
            self.p_factor = penalty_type_kwargs.get('p_factor', PenaltyFunction.DEFAULT_P)
            self.base_penalty: Callable = self._lp_norm_penalty
        elif penalty_type.name == PenaltyFunctionTypes.exponential.name:
            self.base_penalty: Callable = self._exponantial_penalty
        else:
            raise NotImplementedError(f'The given type ({penalty_type}) is not implemented.')
        
    def derivative(self, x):
        if self.penalty_type.name == PenaltyFunctionTypes.lp_norm.name:
            return self._lp_norm_derivative(x)
        else:
            raise NotImplementedError()
      
    def __call__(self, x):
        return (1 / self.gamma) * self.base_penalty(self.gamma * x)
        
    def _lp_norm_penalty(self, x):
        return 1 / self.p_factor * torch.pow(
            torch.relu(x), self.p_factor
        )
    
    @staticmethod
    def _exponantial_penalty(x):
        return torch.exp(x - 1)
    
    def _lp_norm_derivative(self, x):
        return torch.pow(
            self.gamma * torch.relu(x), self.p_factor - 1
        )