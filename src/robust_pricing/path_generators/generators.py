from typing import Union, Optional

from ._base import _PathGenerator
import torch


class Gaussian(_PathGenerator):
    def __init__(
            self,
            path_length,
            mean: Optional[Union[torch.Tensor, int, float]] = None,
            variance: Optional[Union[torch.Tensor, int, float]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if mean is None:
            mean = torch.zeros(path_length)
        elif isinstance(mean, float) or isinstance(mean, int):
            mean = mean * torch.ones(path_length)
        
        self.mean = mean
        
        if variance is None:
            variance = torch.eye(path_length)
        elif isinstance(variance, float) or isinstance(variance, int):
            variance = variance * torch.eye(path_length)
        
        self.variance = variance
        if len(variance) > 1:
            self.sigma = Gaussian.sqrtm(self.variance)
        else:
            self.sigma = torch.sqrt(self.variance)
        
        self.path_length = path_length
    
    @staticmethod
    def sqrtm(matrix: torch.Tensor) -> torch.Tensor:
        from scipy.linalg import sqrtm
        numpy_mat = matrix.detach().numpy()
        
        return torch.Tensor(sqrtm(numpy_mat))
    
    def __call__(self, number_of_samples, *args, **kwargs):
        
        standard_normal = torch.randn(number_of_samples, self.path_length)
        return torch.matmul(standard_normal, self.sigma) + self.mean


class GaussianMartingale(_PathGenerator):
    def __init__(self, path_length, mean: float = 0, variance: float = 1, *args, **kwargs):
        self.path_length = path_length
        self.mean = mean
        self.variance = variance
        super().__init__(*args, **kwargs)
    
    def __call__(self, number_of_samples, *args, **kwargs):
        acum = Gaussian(
            path_length=1, mean=self.mean, variance=self.variance
        )(number_of_samples=number_of_samples)
        
        time_t = acum
        for t in range(self.path_length - 1):
            gaussian = Gaussian(path_length=1, mean=time_t, variance=self.variance)
            time_t = gaussian(number_of_samples=number_of_samples)
            
            acum = torch.cat([acum, time_t], 1)
        
        return acum


class Uniform(_PathGenerator):
    def __init__(
            self,
            path_length,
            mean: Optional[Union[torch.Tensor, int, float]] = None,
            variance: Optional[Union[torch.Tensor, int, float]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if mean is None:
            mean = torch.zeros(path_length)
        elif isinstance(mean, float) or isinstance(mean, int):
            mean = mean * torch.ones(path_length)
        
        self.mean = mean
        
        if variance is None:
            variance = torch.eye(path_length)
        elif isinstance(variance, float) or isinstance(variance, int):
            variance = variance * torch.eye(path_length)
        
        self.variance = variance
        self.sigma = Gaussian.sqrtm(self.variance)
        
        self.path_length = path_length
    
    def __call__(self, number_of_samples, *args, **kwargs):
        
        copula = torch.rand(number_of_samples, self.path_length) - 0.5
        
        return torch.matmul(copula, self.sigma) + self.mean


class UniformMartingale(_PathGenerator):
    def __init__(self, path_length, mean: float = 0, variance: float = 1, *args, **kwargs):
        self.path_length = path_length
        self.mean = mean
        self.variance = variance
        super().__init__(*args, **kwargs)
    
    def __call__(self, number_of_samples, *args, **kwargs):
        acum = Uniform(
            path_length=1, mean=self.mean, variance=self.variance
        )(number_of_samples=number_of_samples)
        
        time_t = acum
        for t in range(self.path_length - 1):
            uniform = Uniform(path_length=1, mean=time_t, variance=self.variance)
            time_t = uniform(number_of_samples=number_of_samples)
            
            acum = torch.cat([acum, time_t], 1)
        
        return acum


class BinomialTree(_PathGenerator):
    def __init__(self, path_length, mean, up_factor, down_factor, granularity, observed_times=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_length = path_length
        self.mean = mean
        self.up_factor = up_factor
        self.down_factor = down_factor
        self.granularity = granularity
        
        self.probability_of_up = (1 - self.down_factor) / (self.up_factor - self.down_factor)
        
        if observed_times is None:
            self.observed_times = [(granularity // path_length) * i % granularity for i in range(1, path_length + 1)]
        else:
            assert len(observed_times) == path_length
            self.observed_times = observed_times

    def __call__(self, number_of_samples, *args, **kwargs):
        
        # Generate random paths (1 model ups in the graph, 0 model downs)
        base = torch.bernoulli(
            torch.ones(number_of_samples, self.granularity) * self.probability_of_up
        )

        up_down_sequence = self.up_factor * base + self.down_factor * (1 - base)

        tensor = None
        for t in self.observed_times:
            new = self.mean * up_down_sequence[::, :t].prod(1).reshape(number_of_samples, 1)
            if tensor is None:
                tensor = new
            else:
                tensor = torch.cat([tensor, new], 1)
        
        return tensor

        