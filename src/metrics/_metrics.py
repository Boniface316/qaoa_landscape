from typing import Tuple, Union

import numpy as np
import orqviz

from ..utils import _isolate_significant_freqs, _truncate_to_only_positive_frequencies


def roughness_tv(
    J: orqviz.aliases.LossFunction,
    theta: orqviz.aliases.ParameterVector,
    M: int,
    m: int,
    l: Union[float, np.ndarray] = np.pi,
) -> Tuple[float, float, float]:
    """Roughness index based on total variation.

    This method assumes that `l` represents 1 period of the function.

    Args:
        J: loss function
        theta: parameter vector of the origin
        M: number of directions along which to sample variation
        m: number of steps to take along each direction
        l: length of domain either the same number for each direction
            or a vector of length theta.size() for each basis direction separately.

    From the paper: https://arxiv.org/abs/2103.11069, Section 3
    """
    # 1. Sample M gaussian random directions with mean=0 and variance=1
    d = np.random.normal(0, 1, size=(M, theta.size))
    # 2. Filter-wise normalizations (section 2.3)
    # make each vector have length 1
    for i in range(M):
        d[i] = d[i] / np.linalg.norm(d[i])

    T = np.zeros(M)
    T_globally_normalized = np.zeros(M)
    min_global_value = np.inf
    max_global_value = -np.inf
    for i in range(M):
        # Calculate the length in each direction such that a single period is not
        # exceeded in the sample. (I'm not sure if this is the right way to normalize)
        l_i = l if isinstance(l, float) else np.min(l / np.abs(d[i]))
        # 3.
        s = np.linspace(-l_i, l_i, m + 1)
        eval = [J(theta + s[j] * d[i]) for j in range(m + 1)]
        rng = max(eval) - min(eval)
        T_i = 0
        for j in range(m):
            T_i += np.abs(eval[j] - eval[j + 1])
        # Normalize by range
        T[i] = T_i / rng
        if max(eval) > max_global_value:
            max_global_value = max(eval)
        if min(eval) < min_global_value:
            min_global_value = min(eval)
        T_globally_normalized[i] = T_i
        # This method assumes that the domain is 1 period of the function so we don't
        # normalize by domain. (domain = 2 * l_i)
        # But if we were to, I found that dividing by `np.sqrt(domain/period)` yields
        # more consistent results
    global_rng = max_global_value - min_global_value
    T_globally_normalized = T_globally_normalized / global_rng
    # The original paper used np.std(T) / np.mean(T) but preliminary testing found that
    # just using the mean yields more consistent results (but perhaps don't do this)
    # return np.mean(T)
    T_value = np.mean(T)
    T_with_std = np.std(T) / np.mean(T)
    T_value_global_norm = np.mean(T_globally_normalized)
    T_with_std_global_norm = np.std(T_globally_normalized) / np.mean(
        T_globally_normalized
    )
    return T_value, T_with_std, T_value_global_norm, T_with_std_global_norm


def roughness_fourier_enrico_2d(
    fourier_result: orqviz.fourier.FourierResult,
) -> Tuple[float, float]:
    """Eq 16 in the following document
    https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9a36d276-5f9a-40df-91ea-b177c81f0d88/Total_variation_multidimensional.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220830%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220830T184459Z&X-Amz-Expires=86400&X-Amz-Signature=faba88510dc663763b976b678d9cabd968ab14a5a06aaf245b29e67a6581e129&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Total_variation_multidimensional.pdf%22&x-id=GetObject

    This method is robust to number of dimensions but right now only implemented on 2d
    because Laura is lazy (& it fits the orqviz api more easily)
    """
    # v = _truncate_to_only_positive_frequencies(fourier_result.values)
    v = fourier_result.values
    assert v.shape[0] == v.shape[1]
    max_k_plus_1 = v.shape[0]
    k_x = np.array([np.arange(max_k_plus_1) for _ in range(max_k_plus_1)])
    k_y = k_x.T

    # rescale k to the actual frequency on the 0-2pi domain.
    # norm_x = (fourier_result.end_points_x[1] - fourier_result.end_points_x[0]) / (
    #     2 * np.pi
    # )
    # norm_y = (fourier_result.end_points_y[1] - fourier_result.end_points_y[0]) / (
    #     2 * np.pi
    # )
    # k_x = k_x / norm_x
    # k_y = k_y / norm_y

    k = np.array([k_x, k_y]).T
    k_norm = np.linalg.norm(k, axis=2)
    k_norm_times_c_k = k_norm * v

    # normalize by the largest fourier coefficient
    # this is to make it so that the roughness of 3*sin(x) is the same as the roughness
    # of sin(x)
    k_norm_times_c_k = np.abs(k_norm_times_c_k)
    k_norm_times_c_k = k_norm_times_c_k / np.max(np.abs(fourier_result.values))

    # threshold
    k_norm_times_c_k = np.where(np.abs(k_norm_times_c_k) < 1e-7, 0, k_norm_times_c_k)

    max_metric = np.max(k_norm_times_c_k)
    sum_metric = np.sum(k_norm_times_c_k) / max_k_plus_1
    return (max_metric, sum_metric)


def roughness_fourier(
    fourier_result: orqviz.fourier.FourierResult, scan_resolution: int = None
) -> float:
    """Roughness index based on Fourier coefficients.

    Assumes `scan_resolution` is the same for all directions.
    Assumes `scan_resolution` is odd, so that the Fourier frequencies are symmetric.
        (This roughness index could be implemented for even resolutions too, but it's
        just not in this current method right now because Laura doesn't fully understand
        what the middle term in a Fourier transform with even resolution means.)

    Basically, the idea is that f(k) * k for any k is bound by total variation AND the
    norm of a function, according to https://arxiv.org/abs/2103.11069 and
    https://math.stackexchange.com/q/68910 and some other sources. Thus, I thought
    sum(f(k) * k for all k) would function somewhat like a roughness index.
    """
    if scan_resolution is None:
        scan_resolution = fourier_result.values.shape[0]
    res = scan_resolution
    if (res % 2) == 0:
        raise ValueError(
            "scan_resolution must be odd for the implementation of this method to work"
        )

    variation_along_x = np.array([list(range(res // 2 + 1))] * res)
    variation_along_y = np.array(
        [list(range(-res // 2 + 1, res // 2 + 1))] * (res // 2 + 1)
    ).T

    # normalize frequencies for range bigger than 1 period of 2pi
    # norm_x = (fourier_result.end_points_x[1] - fourier_result.end_points_x[0]) / (
    #     2 * np.pi
    # )
    # norm_y = (fourier_result.end_points_y[1] - fourier_result.end_points_y[0]) / (
    #     2 * np.pi
    # )
    # variation_along_x = variation_along_x / norm_x
    # variation_along_y = variation_along_y / norm_y

    unnormalized_roughness = (
        np.abs(variation_along_x * variation_along_y) * np.abs(fourier_result.values)
    ).sum()
    return unnormalized_roughness / res / np.max(np.abs(fourier_result.values))
    # Although it's normalized by resolution, it's still best to use the same resolution
    # for valid comparison because I'm not entirely sure that the scaling factor is
    # correct.


def roughness_norm(
    J: orqviz.aliases.LossFunction,
    theta: np.ndarray,
    m: int,
    l_x: float = np.pi,
    l_y: float = np.pi,
) -> float:
    """Roughness index based on the norm of the loss function (how different it is from
    f(x)=0).

    Assumes input cost function takes a 2-D parameter vector as input.

    This is defined as Rq on this page https://en.wikipedia.org/wiki/Surface_roughness
    for 1-D functions.

    """
    # default from -pi to pi
    s = np.linspace(-l_x, l_x, m + 1)
    s_y = np.linspace(-l_y, l_y, m + 1)

    eval = np.array(
        [
            [J(theta + np.array([s[i], s_y[j]])) for i in range(m + 1)]
            for j in range(m + 1)
        ]
    )
    domain = 2 * l_x * 2 * l_y
    rng = np.max(eval) - np.min(eval)
    # q: does the order of normalization by common factor and square/sum/sqrt matter?
    normalized_arr = (eval - np.mean(eval)) / (domain * rng)
    return np.sqrt(np.square(normalized_arr).sum())


def roughness_norm_from_scan_result(
    scan2D_result: orqviz.scans.Scan2DResult,
    domain_size_x: float,  # combines direction vectors of scan and end points
    domain_size_y: float,  # combines direction vectors of scan and end points
) -> float:
    """Roughness index based on the norm of the loss function (how different it is from
    f(x)=0).

    Assumes input cost function takes a 2-D parameter vector as input.

    This is defined as Rq on this page https://en.wikipedia.org/wiki/Surface_roughness
    for 1-D functions.

    """
    eval = scan2D_result.values
    domain = 2 * domain_size_x * 2 * domain_size_y
    rng = np.max(eval) - np.min(eval)
    # q: does the order of normalization by common factor and square/sum/sqrt matter?
    normalized_arr = (eval - np.mean(eval)) / (domain * rng)
    return np.sqrt(np.square(normalized_arr).sum())


def roughness_fourier_sparsity(
    fourier_result: orqviz.fourier.FourierResult, threshold_factor=0.1
) -> float:
    # threshold = 1e-2
    # significant_freqs = 1 - np.isclose(
    #     fourier_result.values, 0, atol=threshold
    # )
    significant_freqs = _isolate_significant_freqs(
        fourier_result.values, threshold_factor
    )
    significant_freqs[0][0] = 0  # don't count the constant term
    return np.count_nonzero(significant_freqs)


def roughness_fourier_sparsity_using_norms(
    fourier_result: orqviz.fourier.FourierResult,
) -> float:
    one_norm = np.sum(np.abs(fourier_result.values))
    # two_norm = np.sum(np.abs(np.square(fourier_result.values)))
    vector_fourier_result = fourier_result.values.reshape(-1)
    two_norm = np.linalg.norm(vector_fourier_result)
    return one_norm**2 / two_norm**2


def roughness_tv_from_scan(scan2D_result):
    # Perhaps the final value should be divided by 2, but we couldn't get
    # a good grasp of it, and in the end decided it's not blocking for now,
    # as it's just a constant multiplier and it won't change the behaviour
    # of the metric in relation between various cost funcitons.
    abs_difference_x_arr = np.abs(
        scan2D_result.values[0:-1, :] - scan2D_result.values[1:, :]
    )
    abs_difference_y_arr = np.abs(
        scan2D_result.values[:, 1:-1] - scan2D_result.values[:, 0:-2]
    )
    values_diff = np.max(scan2D_result.values) - np.min(scan2D_result.values)

    integral_x = np.sum(abs_difference_x_arr / scan2D_result.values.shape[0])
    integral_y = np.sum(abs_difference_y_arr / scan2D_result.values.shape[1])

    return (integral_x + integral_y) / values_diff
