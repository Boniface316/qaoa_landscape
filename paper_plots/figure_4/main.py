import os
import sys
from typing import Dict

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from orquestra.integrations.qulacs import QulacsSimulator
from orquestra.quantum.operators import PauliSum, PauliTerm

from src.data_generation import (
    generate_data,
    load_data,
    prepare_cost_function,
    save_hamiltonians,
)
from src.hamiltonian_generation import all_linear_terms
from src.plots import create_plot
from src.utils import calculate_plot_extents, generate_timestamp_str

backend = QulacsSimulator()
scan_resolution = 201


def main():
    cost_period = [np.pi, np.pi]
    fourier_period = [2 * np.pi, 2 * np.pi]
    hamiltonians_dict: Dict[str, PauliSum] = {}

    # CASE 1: All linear terms + one term of higher order
    num_qubits = 6
    hamiltonians_dict["ham_0"] = all_linear_terms(num_qubits)

    hamiltonians_dict["ham_A"] = all_linear_terms(num_qubits) + PauliTerm("Z0")

    hamiltonians_dict["ham_B"] = all_linear_terms(num_qubits) + PauliTerm(
        "Z0"
    ) * PauliTerm("Z1")

    hamiltonians_dict["ham_C"] = all_linear_terms(num_qubits) + PauliTerm(
        "Z0"
    ) * PauliTerm("Z1") * PauliTerm("Z2")

    hamiltonians_dict["ham_D"] = all_linear_terms(num_qubits) + PauliTerm(
        "Z0"
    ) * PauliTerm("Z1") * PauliTerm("Z2") * PauliTerm("Z3")

    hamiltonians_dict["ham_E"] = all_linear_terms(num_qubits) + PauliTerm(
        "Z0"
    ) * PauliTerm("Z1") * PauliTerm("Z2") * PauliTerm("Z3") * PauliTerm("Z4")

    hamiltonians_dict["ham_F"] = all_linear_terms(num_qubits) + PauliTerm(
        "Z0"
    ) * PauliTerm("Z1") * PauliTerm("Z2") * PauliTerm("Z3") * PauliTerm(
        "Z4"
    ) * PauliTerm(
        "Z5"
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
            fourier_res_x=12,
            fourier_res_y=14,
            timestamp_str=timestamp,
            unit="pi",
            include_all_metrics=False,
            remove_constant=True,
        )


if __name__ == "__main__":
    main()
