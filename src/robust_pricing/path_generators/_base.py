from abc import abstractmethod


class _PathGenerator:
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def __call__(self, number_of_samples, *args, **kwargs):
        ...
