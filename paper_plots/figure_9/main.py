import os
import sys

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.ticker as tck
import numpy as np
import orqviz
from matplotlib import pyplot as plt
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.quantum.operators._pauli_operators import PauliTerm
from orquestra.quantum.utils import create_symbols_map
from orquestra.vqa.ansatz.qaoa_farhi import QAOAFarhiAnsatz
from orqviz.plot_utils import get_colorbar_from_ax

from src.utils import generate_timestamp_str

interesting_hamiltonian = PauliTerm("Z0") + PauliTerm("Z1", 2) + PauliTerm("Z0*Z2", 3)
sim = QulacsSimulator()
n_layers = 1
params = np.random.uniform(-np.pi, np.pi, size=n_layers * 2)
end_points = (0, 2 * np.pi)
fourier_res = 10
dir1 = np.array([1.0, 0.0])
dir2 = np.array([0.0, 1.0])
resolutions = [20, 50, 100]
timestamp_str = generate_timestamp_str()

results_directory = os.path.join(os.getcwd(), f"results/plots_{timestamp_str}")
os.makedirs(results_directory, exist_ok=True)


fig, axes = plt.subplots(2, 3, figsize=(23, 10))


def loss_function(pars: np.ndarray) -> float:
    ansatz = QAOAFarhiAnsatz(n_layers, interesting_hamiltonian)
    circuit = ansatz.parametrized_circuit
    circuit_symbols = circuit.free_symbols

    symbols_map = create_symbols_map(circuit_symbols, pars)
    circuit = circuit.bind(symbols_map)
    return sim.get_exact_expectation_values(circuit, interesting_hamiltonian)


for i, resolution in enumerate(resolutions):
    scan2D_result = orqviz.scans.perform_2D_scan(
        params,
        loss_function,
        direction_x=dir1,
        direction_y=dir2,
        n_steps_x=resolution,
        end_points_x=end_points,
    )

    fourier_result = orqviz.fourier.perform_2D_fourier_transform(
        scan2D_result, end_points, end_points
    )
    ax1 = axes[0, i]
    ax2 = axes[1, i]

    orqviz.scans.plot_2D_scan_result(scan2D_result, ax=ax1)
    ax1.set_title(
        f"Original function, scanned with resolution {resolution}", fontsize=25
    )

    orqviz.fourier.plot_2D_fourier_result(
        fourier_result, max_freq_x=fourier_res, fig=fig, ax=ax2
    )
    ax1.xaxis.set_major_locator(tck.MultipleLocator(base=np.pi))
    ax1.yaxis.set_major_locator(tck.MultipleLocator(base=np.pi))
    ax2.set_title("Fourier transform of the above landscape", fontsize=25)
    ax2.tick_params(axis="both", labelsize=25)
    ax1.tick_params(axis="both", labelsize=25)
    ax1.title.set_size(18)
    ax2.title.set_size(18)
    ax1.set_xlabel("$\\gamma$", fontsize=18)
    ax1.set_ylabel("$\\beta$", fontsize=18)
    ax2.set_xlabel("$f_{\\gamma}$", fontsize=18)
    ax2.set_ylabel("$f_{\\beta}$", fontsize=18)

    ax1.xaxis.set_major_formatter(
        tck.FuncFormatter(
            lambda val, pos: "{:.2f}$\pi$".format(val / np.pi) if val != 0 else "0"
        )
    )
    ax1.yaxis.set_major_formatter(
        tck.FuncFormatter(
            lambda val, pos: "{:.2f}$\pi$".format(val / np.pi) if val != 0 else "0"
        )
    )

    ax1.tick_params(axis="both", labelsize=15)
    ax2.tick_params(axis="both", labelsize=15)

    cbar = get_colorbar_from_ax(ax1)
    cbar.ax.tick_params(labelsize=15)

    cbar2 = get_colorbar_from_ax(ax2)
    cbar2.ax.tick_params(labelsize=15)


plt.tight_layout()
plt.savefig(f"{results_directory}/resolution-artifacts_2.png")
plt.clf()
plt.close()
