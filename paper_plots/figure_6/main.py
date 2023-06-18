import math
import os
import sys
from typing import Dict

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from orquestra.integrations.qulacs import QulacsSimulator
from orquestra.quantum.operators import PauliSum

from src.data_generation import (
    generate_data,
    load_data,
    prepare_cost_function,
    save_hamiltonians,
)
from src.hamiltonian_generation import all_cubic_terms, all_linear_terms
from src.plots import create_plot
from src.utils import calculate_plot_extents, generate_timestamp_str

backend = QulacsSimulator()
scan_resolution = 201


def main():
    all_metrics = []
    all_names = []
    cost_period = [np.pi, np.pi]
    fourier_period = [2 * np.pi, 2 * np.pi]
    num_qubits = 6
    hamiltonians_dict: Dict[str, PauliSum] = {}

    base_hamiltonian = all_linear_terms(num_qubits)
    hamiltonians_dict["base_hamiltonian"] = base_hamiltonian
    all_cubic = all_cubic_terms(num_qubits)

    for i in range(len(all_cubic.terms)):
        hamiltonians_dict[f"ham_basic_{i}"] = base_hamiltonian + PauliSum(
            all_cubic.terms[: i + 1]
        )

    timestamp = generate_timestamp_str()

    save_hamiltonians(hamiltonians_dict)

    for file_label, hamiltonian in hamiltonians_dict.items():
        cost_function = prepare_cost_function(hamiltonian, backend)
        origin = np.array([0, 0])
        dir_x = np.array([1, 0])
        dir_y = np.array([0, 1])

        period, fourier_res_x, fourier_res_y = calculate_plot_extents(hamiltonian)

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

        all_metrics.append(metrics_dict)
        all_names.append(file_label)

        create_plot(
            scan2D_result,
            fourier_result,
            metrics_dict,
            label=file_label,
            fourier_res_x=fourier_res_x,
            fourier_res_y=fourier_res_y,
            timestamp_str=timestamp,
            remove_constant=True,
            unit="pi",
            include_all_metrics=False,
        )

    tvs = [el["tv"] for el in all_metrics]
    fouriers = [el["fourier density using norms"] for el in all_metrics]

    plots_folder = os.path.join(os.getcwd(), f"results/plots_{timestamp}")
    plt.plot(tvs)
    plt.savefig(f"{plots_folder}/tv_change_plot.png")
    plt.clf()

    plt.plot(fouriers)
    plt.savefig(f"{plots_folder}/fourier_change_plot.png")
    plt.clf()


if __name__ == "__main__":
    main()
