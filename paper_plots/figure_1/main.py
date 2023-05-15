import os
import sys
from typing import Dict

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.quantum.operators import PauliSum, PauliTerm

from src.data_generation import generate_data, prepare_cost_function
from src.plots import create_plot
from src.utils import calculate_plot_extents, generate_timestamp_str

backend = QulacsSimulator()
scan_resolution = 101


def main():
    hamiltonians_dict: Dict[str, PauliSum] = {}

    hamiltonians_dict["ham_1"] = (
        -2.75 * PauliTerm("Z0") - 3.25 * PauliTerm("Z1") + 3.75 * PauliTerm("Z0*Z1")
    )

    hamiltonians_dict["ham_2"] = (
        0.75 * PauliTerm("Z1*Z0")
        + 0.75 * PauliTerm("Z2*Z3")
        + 0.75 * PauliTerm("Z4*Z5")
        + 0.25 * PauliTerm("Z1")
        + 0.25 * PauliTerm("Z3")
        - 0.125 * PauliTerm("Z3*Z1")
        + 0.25 * PauliTerm("Z0")
        + 0.25 * PauliTerm("Z2")
        - 0.125 * PauliTerm("Z2*Z0")
        + 0.125 * PauliTerm("Z1*Z2")
        - 0.125 * PauliTerm("Z1*Z2*Z0")
        + 0.125 * PauliTerm("Z3*Z0")
        - 0.125 * PauliTerm("Z3*Z2*Z0")
        - 0.125 * PauliTerm("Z3*Z1*Z0")
        - 0.125 * PauliTerm("Z3*Z1*Z2")
        + 0.125 * PauliTerm("Z3*Z1*Z2*Z0")
        + 0.25 * PauliTerm("Z5")
        - 0.125 * PauliTerm("Z2*Z5")
        + 0.25 * PauliTerm("Z4")
        - 0.125 * PauliTerm("Z3*Z4")
        + 0.125 * PauliTerm("Z5*Z3")
        - 0.125 * PauliTerm("Z5*Z3*Z4")
        + 0.125 * PauliTerm("Z2*Z4")
        - 0.125 * PauliTerm("Z2*Z3*Z4")
        - 0.125 * PauliTerm("Z2*Z5*Z4")
        - 0.125 * PauliTerm("Z2*Z5*Z3")
        + 0.125 * PauliTerm("Z2*Z5*Z3*Z4")
        - 0.125 * PauliTerm("Z1*Z5")
        - 0.125 * PauliTerm("Z0*Z4")
        + 0.125 * PauliTerm("Z5*Z0")
        - 0.125 * PauliTerm("Z5*Z0*Z4")
        + 0.125 * PauliTerm("Z1*Z4")
        - 0.125 * PauliTerm("Z1*Z0*Z4")
        - 0.125 * PauliTerm("Z1*Z5*Z4")
        - 0.125 * PauliTerm("Z1*Z5*Z0")
        + 0.125 * PauliTerm("Z1*Z5*Z0*Z4")
    )

    timestamp = generate_timestamp_str()
    for file_label, hamiltonian in hamiltonians_dict.items():
        cost_function = prepare_cost_function(hamiltonian, backend)
        origin = np.array([0, 0])
        dir_x = np.array([1, 0])
        dir_y = np.array([0, 1])

        cost_period = [2 * np.pi, 2 * np.pi]
        fourier_period = [2 * np.pi, 2 * np.pi]

        _, fourier_res_x, fourier_res_y = calculate_plot_extents(hamiltonian)

        scan2D_result, fourier_result, metrics_dict = generate_data(
            cost_function=cost_function,
            origin=origin,
            dir_x=dir_x,
            dir_y=dir_y,
            file_label=file_label,
            scan_resolution=scan_resolution,
            cost_period=cost_period,
            fourier_period=fourier_period,
            timestamp_str=timestamp,
        )

        create_plot(
            scan2D_result=scan2D_result,
            fourier_result=fourier_result,
            metrics_dict=metrics_dict,
            label=file_label,
            fourier_res_x=fourier_res_x,
            fourier_res_y=fourier_res_y,
            timestamp_str=timestamp,
            unit="pi",
            remove_constant=True,
            include_all_metrics=False,
        )


if __name__ == "__main__":
    main()
