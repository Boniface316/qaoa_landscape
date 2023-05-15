# see experiments/Enrico_experiments/total_variation.ipynb

import numpy as np
from numdifftools.core import (  # Quite useful for automating finite differences gradient calculation. Replace with other methods as needed.
    Gradient,
)

tau = np.pi * 2


def g(f, param, l, p=1):
    # Calculates gradient using finite differences
    grad = Gradient(f)
    return np.linalg.norm(grad(param) * l) ** p


def sampler(l=np.array([tau, tau])):
    # param sampler
    # Defines how the samples are taken, in this case uniformly across the domain [0, l]. Default period is 2/pi
    while True:
        yield np.random.uniform(0, l)


def integrate(integrand, sampler, n):
    samples = [integrand(next(sampler)) for _ in range(n)]
    rng = np.max(samples) - np.min(samples)
    return sum(samples) / (n * rng)


def tv_numerical_integration(f, n=200, p=1, l=np.array([tau, tau])):
    """
    f: function to find TV of
    n: number of samples
    p: Laura doesn't fully understand tbh
    l: length of domain along each dimension (default is 2pi)
    """
    integrand = lambda x: g(f, x, l=l, p=p)
    tv = integrate(integrand, sampler(), n=n)

    return tv ** (1 / p)
