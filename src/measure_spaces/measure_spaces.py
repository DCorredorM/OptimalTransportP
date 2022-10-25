from abc import abstractmethod
from collections.abc import Iterable

import numpy as np
import scipy.stats as sts


##################################################
################# Measure Spaces #################
##################################################


class MeasureSpace:
    def __init__(self, measure: 'Measure' = None):
        self.measure = measure if measure is not None else self._default_measure()

    @abstractmethod
    def is_measurable(self, subset):
        ...

    @abstractmethod
    def integrate(self, function):
        ...

    @abstractmethod
    def _default_measure(self):
        ...


class DiscreteMeasureSpace(MeasureSpace):
    def __init__(self,
                 support: list,
                 measure: 'DiscreteMeasure' = None,
                 ):
        self.support = support
        super().__init__(measure)

    def is_measurable(self, subset):
        if subset in self.support or set(self.support).issuperset(subset):
            return True
        return False

    def integrate(self, function):
        return sum(function(x) * self.measure(x) for x in self.support)

    @abstractmethod
    def _default_measure(self):
        mass = [1 / len(self.support)] * len(self.support)
        return DiscreteMeasure(list(self.support), mass, self)

##################################################
#################### Measures ####################
##################################################


class Measure:
    def __init__(self, measure_space: 'MeasureSpace' = None, check=False):
        self.measure_space = measure_space if measure_space is not None else self._default_measure_space()
        if check:
            self._check_measure()

    def __call__(self, measurable):
        return self._measure(measurable)

    @abstractmethod
    def _measure(self, measurable):
        ...

    @abstractmethod
    def _density(self, point):
        ...

    @abstractmethod
    def _default_measure_space(self):
        ...

    @abstractmethod
    def _check_measure(self):
        ...


class DiscreteMeasure(Measure):
    def __init__(self, support: list, mass: list= None, density=None, measure_space: 'MeasureSpace' = None, index_support=True):
        if density is not None:
            if isinstance(density, sts._distn_infrastructure.rv_frozen):
                if index_support:
                    mass = [density.pmf(i) for i in range(len(support))]
                else:
                    mass = [density.pmf(s) for s in support]
                
        self.support = support
        self.mass = mass
        self.density = lambda x: mass[support.index(x)]

        super().__init__(measure_space)

    @classmethod
    def from_histogram(cls, histogram: dict, measure_space: 'MeasureSpace' = None):
        support = list(histogram.keys())
        mass = list(histogram.values())
        return cls(support, mass, measure_space)

    @classmethod
    def from_scipy_continuous(cls, scipy_continuous, bins=None, measure_space: 'MeasureSpace' = None):
        # TODO: Fix this method
        n = 1000
        support, mass = np.histogram(scipy_continuous.rvs(n), bins=bins)
        support = (support[:-1] + support[1:]) / 2
        mass = mass / n
        return cls(support, mass, measure_space)

    def _measure(self, measurable):
        if isinstance(measurable, Iterable):
            return sum(self._density(point) for point in measurable)
        else:
            return self._density(measurable)

    def _density(self, point):
        if point in self.support:
            return self.density(point)
        return 0

    def _default_measure_space(self):
        return DiscreteMeasureSpace(self.support, self)

    def _check_measure(self):
        _sum = sum(self.mass)
        if len(self.support) != len(self.mass):
            raise Exception('You need to provide a valid measure.\n'
                            'The support and mass lists that you provided differ from length.\n'
                            f'support list has length {len(self.support)}\n'
                            f'mass list has length {len(self.mass)}.')

        if _sum == 1:
            return True
        else:
            self.mass = list(map(lambda x: x / _sum, self.mass))
            return True


