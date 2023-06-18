# see experiments/Enrico_experiments/total_variation.ipynb

from typing import Callable, Iterator, List

import numpy as np
from numdifftools.core import (
    Gradient,  # Quite useful for automating finite differences gradient calculation. Replace with other methods as needed.
)

tau = np.pi * 2


def g(
    f: Callable[[np.ndarray], float], param: np.ndarray, l: np.ndarray, p: int = 1
) -> float:
    # Calculates gradient using finite differences
    grad = Gradient(f)
    return np.linalg.norm(grad(param) * l) ** p


def sampler(l: np.array([tau, tau])) -> float:
    while True:
        yield np.random.uniform(0, l)


def integrate(
    integrand: Callable[[float], float], sampler: Iterator[float], n: int
) -> float:
    samples: List[float] = [integrand(next(sampler)) for _ in range(n)]
    rng: float = np.max(samples) - np.min(samples)
    return sum(samples) / (n * rng)


def tv_numerical_integration(
    f: Callable[[np.ndarray], float],
    n: int = 200,
    p: int = 1,
    l: np.ndarray = np.array([tau, tau]),
) -> float:
    """
    f: function to find TV of
    n: number of samples
    p: Laura doesn't fully understand tbh
    l: length of domain along each dimension (default is 2pi)
    """
    integrand = lambda x: g(f, x, l=l, p=p)
    tv = integrate(integrand, sampler(), n=n)

    return tv ** (1 / p)
