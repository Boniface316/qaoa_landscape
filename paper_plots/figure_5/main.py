import math
import os
import sys
from typing import Dict

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from orquestra.integrations.qulacs import QulacsSimulator
from orquestra.quantum.operators import PauliSum, PauliTerm

from src.data_generation import generate_data, prepare_cost_function, save_hamiltonians
from src.hamiltonian_generation import (
    all_cubic_terms,
    all_linear_terms,
    all_quadratic_terms,
    assign_weight_for_term,
)
from src.plots import create_plot
from src.utils import calculate_plot_extents, generate_timestamp_str

backend = QulacsSimulator()
scan_resolution = 401


def main():
    cost_period = [np.pi, np.pi]
    fourier_period = [2 * np.pi, 2 * np.pi]
    num_qubits = 6
    hamiltonians_dict: Dict[str, PauliSum] = {}

    temp_hamiltonian = (
        all_linear_terms(num_qubits)
        + all_quadratic_terms(num_qubits)
        + all_cubic_terms(num_qubits)
    )

    hamiltonians_dict["ham_A_plus_B_plus_C"] = temp_hamiltonian

    temp_hamiltonian = (
        all_linear_terms(num_qubits)
        + all_quadratic_terms(num_qubits)
        + all_cubic_terms(num_qubits)
    )

    coeff = 25
    hamiltonians_dict[f"ham_A_plus_B_plus_C_Z0_{coeff}"] = assign_weight_for_term(
        temp_hamiltonian, PauliTerm("Z0"), coeff + 0j
    )
    temp_hamiltonian = (
        all_linear_terms(num_qubits)
        + all_quadratic_terms(num_qubits)
        + all_cubic_terms(num_qubits)
    )

    hamiltonians_dict[f"ham_A_plus_B_plus_C_Z0Z1_{coeff}"] = assign_weight_for_term(
        temp_hamiltonian, PauliTerm("Z0") * PauliTerm("Z1"), coeff + 0j
    )

    temp_hamiltonian = (
        all_linear_terms(num_qubits)
        + all_quadratic_terms(num_qubits)
        + all_cubic_terms(num_qubits)
    )

    hamiltonians_dict[f"ham_A_plus_B_plus_C_Z0Z1Z2_{coeff}"] = assign_weight_for_term(
        temp_hamiltonian,
        PauliTerm("Z0") * PauliTerm("Z1") * PauliTerm("Z2"),
        coeff + 0j,
    )

    timestamp = generate_timestamp_str()
    # os.mkdir("results")
    save_hamiltonians(hamiltonians_dict, save_in_s3=False)
    total_number_of_job = len(hamiltonians_dict)

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
            fourier_res_x=96,
            fourier_res_y=7,
            timestamp_str=timestamp,
            unit="pi",
            remove_constant=True,
            include_all_metrics=False,
        )


if __name__ == "__main__":
    main()
