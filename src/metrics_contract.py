import numpy as np


def cost_function_1(pars):
    return np.sin(pars[0]) + np.cos(2 * pars[1])


def cost_function_higher_freq(pars):
    return np.sin(pars[0]) + np.cos(10 * pars[1])


def cost_function_higher_value_range(pars):
    return cost_function_1(pars) * 3


def validate_output_is_real(roughness_measure):
    assert isinstance(roughness_measure(cost_function_1), (int, float))


def validate_changing_frequency_and_adjusting_period_doesnt_change_roughness(
    roughness_measure,
):
    roughnesses_1 = []
    roughnesses_2 = []
    for _ in range(10):
        roughnesses_1.append(roughness_measure(cost_function_1))
        roughnesses_2.append(roughness_measure(cost_function_higher_freq))
    # For the case where roughness measure is probabilistic
    atol = np.std(roughnesses_1) + np.std(roughnesses_2)
    # For the case where roughness measure is deterministic we set the value as 1%
    # of the smallest result got, in order to avoid test failing in case of 
    # numerical instabilities.
    if np.isclose(atol, 0):
        atol = 0.01 * min(roughnesses_1+roughnesses_2)
    assert np.isclose(np.mean(roughnesses_1), np.mean(roughnesses_2), atol=atol)


def validate_changing_value_range_doesnt_change_roughness(roughness_measure):
    roughnesses_1 = []
    roughnesses_2 = []
    for _ in range(10):
        roughnesses_1.append(roughness_measure(cost_function_1))
        roughnesses_2.append(roughness_measure(cost_function_higher_value_range))
    assert np.isclose(
        np.mean(roughnesses_1),
        np.mean(roughnesses_2),
        atol=np.std(roughnesses_1) + np.std(roughnesses_2),
    )


def validate_variance_not_too_high(roughness_measure):
    roughnesses = []
    for _ in range(10):
        roughness = roughness_measure(cost_function_1)
        roughnesses.append(roughness)
    assert np.std(roughnesses) / np.mean(roughnesses) < 0.1  # SNR < 10%


"""In addition to these contracts, each metric should also have a test validating that
the roughness converges when resolution increases.
"""
METRICS_CONTRACT = [
    validate_output_is_real,
    validate_changing_frequency_and_adjusting_period_doesnt_change_roughness,
    validate_changing_value_range_doesnt_change_roughness,
    validate_variance_not_too_high,
]
