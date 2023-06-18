import datetime
from typing import Callable, Optional, Tuple

import numpy as np
import orqviz
from orquestra.quantum.operators import PauliRepresentation
from scipy.optimize import OptimizeResult


def generate_timestamp_str() -> str:
    """Create a string in the format <YYYY_MM_DD_HH_mm_ss> capturing the current
    timestamp.
    """
    now = datetime.datetime.now()
    YYYY = now.year
    MM, DD, HH, mm, ss = [
        "0" + str(x) if x < 10 else str(x)
        for x in (now.month, now.day, now.hour, now.minute, now.second)
    ]
    return f"{YYYY}_{MM}_{DD}_{HH}_{mm}_{ss}"


def _truncate_to_only_positive_frequencies(result: np.ndarray) -> np.ndarray:
    """Truncates the result to only positive frequencies.

    see https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details
    """
    n_y = result.shape[0]
    return result[: n_y // 2 + 1]


def _isolate_significant_freqs(array: np.ndarray, threshold_factor=0.1) -> np.ndarray:
    threshold = threshold_factor * np.max(np.abs(array))
    significant_freqs = 1 - np.isclose(np.abs(array), 0, atol=threshold)
    return significant_freqs


def calculate_period_using_fourier(
    fourier_scan: orqviz.fourier.FourierResult, b=10
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    significant_freqs = _isolate_significant_freqs(fourier_scan.values)
    lowest_freq = [0, 0]
    for i, row in enumerate(significant_freqs):
        if i > 0 and np.sum(row) > 0:
            lowest_freq[1] = i  # rows are y frequenices
            break
    for i, col in enumerate(significant_freqs.T):
        if i > 0 and np.sum(col) > 0:
            lowest_freq[0] = i
            break
    period = tuple(2 * np.pi / np.array(lowest_freq))
    return tuple(lowest_freq), period  # type: ignore


def calculate_plot_extents(
    hamiltonian: PauliRepresentation,
    period: np.ndarray = np.array([2 * np.pi, 2 * np.pi]),
) -> Tuple[np.ndarray, int, int]:
    """Calculate the period and the resolution of the Fourier transform of the
    Hamiltonian.

    Args:
        hamiltonian: The Hamiltonian to calculate the extents for.
        period: The period of the Hamiltonian.

    Returns:
        A tuple containing the period, maximum Fourier coefficient, and the
        maximum locality.
    """
    # Set custom period if all coefficients are ints
    coefficients = [term.coefficient for term in hamiltonian.terms]
    if all([isint(coefficient) for coefficient in coefficients]):
        gcd = np.gcd.reduce(np.real(coefficients).astype(int))
        period[0] /= gcd

    min_locality = min(
        [len(term.operations) for term in hamiltonian.terms if not term.is_constant]
    )
    max_locality = max(
        [len(term.operations) for term in hamiltonian.terms if not term.is_constant]
    )
    period[1] /= 2 * min_locality
    period[0] /= 2

    maximum_fourier_coefficient = 2 * np.sum(
        np.abs([term.coefficient for term in hamiltonian.terms])
    )

    return period, maximum_fourier_coefficient + 1, 2 * max_locality + 1


def isint(x: complex) -> bool:
    return x.real == x and int(x.real) == x.real


def optimize_cost_function(
    cost_function: Callable,
    initial_params: Optional[np.ndarray] = None,
) -> OptimizeResult:
    is_finished = False
    exclude_borders = True
    while not is_finished and exclude_borders:
        bound_limit = np.pi * 0.9
        bounds = [(-bound_limit, bound_limit), (-bound_limit, bound_limit)]
        if initial_params is None:
            initial_params = np.random.uniform(
                low=bounds[0][0], high=bounds[0][1], size=(2,)
            )
        else:
            exclude_borders = False
        optimizer = ScipyOptimizer(
            "L-BFGS-B",
            bounds=bounds,
        )
        results = optimizer.minimize(
            cost_function=cost_function, initial_params=initial_params
        )
        # Check if optimization finished successfully
        if np.isclose(np.abs(results.opt_params[0]), np.pi) or np.isclose(
            np.abs(results.opt_params[1]), np.pi
        ):
            is_finished = False
            initial_params = None
        else:
            is_finished = True
    return results
