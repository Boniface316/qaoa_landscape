import math
import os
import sys
from typing import Dict

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.quantum.operators import PauliSum, PauliTerm

from src.data_generation import (
    generate_data,
    load_data,
    prepare_cost_function,
    save_hamiltonians,
)
from src.plots import create_plot
from src.utils import calculate_plot_extents, generate_timestamp_str

backend = QulacsSimulator()
scan_resolution = 201


def main():
    cost_period = [np.pi, np.pi]
    fourier_period = [2 * np.pi, 2 * np.pi]
    hamiltonians_dict: Dict[str, PauliSum] = {}

    hamiltonians_dict["ham_6_1_23_4_47_6_05"] = (
        1.23 * PauliTerm("Z0") + 4.47 * PauliTerm("Z1") + 6.05 * PauliTerm("Z0*Z1")
    )

    hamiltonians_dict["ham_6_1_4_6"] = (
        1 * PauliTerm("Z0") + 4 * PauliTerm("Z1") + 6 * PauliTerm("Z0*Z1")
    )

    timestamp = generate_timestamp_str()

    save_hamiltonians(hamiltonians_dict)

    for file_label, hamiltonian in hamiltonians_dict.items():
        cost_function = prepare_cost_function(hamiltonian, backend)
        origin = np.array([0, 0])
        dir_x = np.array([1, 0])
        dir_y = np.array([0, 1])

        scan2D_result, fourier_result, metrics_dict = generate_data(
            cost_function,
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
            scan2D_result,
            fourier_result,
            metrics_dict,
            label=file_label,
            fourier_res_x=32,
            fourier_res_y=6,
            timestamp_str=timestamp,
            include_all_metrics=False,
            remove_constant=True,
            unit="pi",
        )


if __name__ == "__main__":
    main()
