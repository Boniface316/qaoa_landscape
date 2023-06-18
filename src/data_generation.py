import json
import operator
import os
from functools import lru_cache, partial, reduce
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import orqviz
from orquestra.quantum.api.backend import QuantumSimulator
from orquestra.quantum.circuits import CNOT, RX, RZ, Circuit, H, create_layer_of_gates
from orquestra.quantum.operators import (
    PauliSum,
    PauliTerm,
    load_operator,
    save_operator,
)
from orqviz.scans import Scan2DResult

from .metrics import roughness_fourier_sparsity_using_norms, roughness_tv
from .utils import generate_timestamp_str

tau = np.pi * 2


@lru_cache(maxsize=256)
def time_evolution_cost(
    hamiltonian: PauliSum, time: int, method: str = "Trotter", trotter_order: int = 1
):
    return reduce(
        operator.add,
        (
            time_evolution_for_term_cost(term, time / trotter_order)
            for _index_order in range(trotter_order)
            for term in hamiltonian.terms
        ),
    )


def time_evolution_for_term(term: PauliTerm, time: int):
    base_changes = []
    base_reversals = []
    cnot_gates = []
    central_gate = None
    qubit_indices = sorted(term.qubits)

    # circuit = circuits.Circuit()
    gates = []
    # If constant term, return empty circuit.
    if term.is_constant:
        return Circuit()
    coefficient = term.coefficient
    for i, qubit_id in enumerate(qubit_indices):
        term_type = term[qubit_id]
        if term_type == "X":
            base_changes.append(H(qubit_id))
            base_reversals.append(H(qubit_id))
        elif term_type == "Y":
            base_changes.append(RX(np.pi / 2)(qubit_id))
            base_reversals.append(RX(-np.pi / 2)(qubit_id))
        if i == len(term.operations) - 1:
            central_gate = RZ(2 * time * coefficient)(qubit_id)
        else:
            cnot_gates.append(CNOT(qubit_id, qubit_indices[i + 1]))

    for gate in base_changes:
        gates.append(gate)

    for gate in cnot_gates:
        gates.append(gate)

    if central_gate is not None:
        gates.append(central_gate)

    for gate in reversed(cnot_gates):
        gates.append(gate)

    for gate in base_reversals:
        gates.append(gate)

    return Circuit(gates)


@lru_cache(maxsize=256)
def time_evolution_for_term_cost(term: PauliTerm, coeff: float):
    return time_evolution_for_term(term, coeff)


@lru_cache(maxsize=256)
def time_evolution_for_term_mixer(term: PauliTerm, coeff: float):
    return time_evolution_for_term(term, coeff)


@lru_cache(maxsize=256)
def time_evolution_mixer(
    hamiltonian: PauliSum, time: int, method: str = "Trotter", trotter_order: int = 1
):
    return reduce(
        operator.add,
        (
            time_evolution_for_term_mixer(term, time / trotter_order)
            for _index_order in range(trotter_order)
            for term in hamiltonian.terms
        ),
    )


def prepare_cost_function(
    operator: PauliSum, backend: QuantumSimulator, n_layers: int = 1
) -> orqviz.aliases.LossFunction:
    """Exact expectation values"""
    number_of_qubits = operator.n_qubits
    mixer_hamiltonian = PauliSum("+".join([f"X{i}" for i in range(number_of_qubits)]))

    def loss_function(
        mixer_op, cost_op, n_qubits, pars: orqviz.aliases.ParameterVector
    ) -> float:
        circuit = Circuit()

        # Prepare initial state
        circuit += create_layer_of_gates(number_of_qubits, H)

        # Add time evolution layers
        circuit += time_evolution_cost(cost_op, pars[0])
        circuit += time_evolution_mixer(mixer_op, pars[1])

        return backend.get_exact_expectation_values(circuit, operator).values.sum()

    return partial(loss_function, mixer_hamiltonian, operator, number_of_qubits)


def generate_data(
    cost_function: orqviz.aliases.LossFunction,
    origin: orqviz.aliases.ParameterVector,
    dir_x: orqviz.aliases.DirectionVector,
    dir_y: orqviz.aliases.DirectionVector,
    file_label: str,  # perhaps a string representing the operator
    scan_resolution: int,
    cost_period: List[float],
    fourier_period: List[float],
    directory_name: str = "data",
    timestamp_str: Optional[str] = None,
    end_points: Tuple[float, float] = (-np.pi, np.pi),
) -> Tuple[Scan2DResult, Scan2DResult, Dict[str, Any]]:
    """
    period: the size of the period in each direction. A 1-d array with the same size
    as the cost function's input parameters.
    custom_plot_period: if we want to make plots with a different period for aesthetic
    purposes. does not impact metrics.

    """

    if timestamp_str is None:
        timestamp_str = generate_timestamp_str()

    directory_name = check_and_create_directory(directory_name, timestamp_str)

    tv = roughness_tv(cost_function, origin, M=200, m=200)[0]

    dir_x = dir_x / np.linalg.norm(dir_x)
    dir_y = dir_y / np.linalg.norm(dir_y)

    scan2D_result = perform_2D_scan(
        cost_function, origin, scan_resolution, dir_x, dir_y, cost_period, end_points
    )

    fourier_result = perform_Fourier_scan(
        cost_function, origin, scan_resolution, dir_x, dir_y, fourier_period, end_points
    )

    metrics_dict = _generate_metrics_dict(scan2D_result, fourier_result, tv)

    save_scan_results(
        scan2D_result, fourier_result, metrics_dict, directory_name, file_label
    )

    return scan2D_result, fourier_result, metrics_dict


def perform_2D_scan(
    cost_function: orqviz.aliases.LossFunction,
    origin: orqviz.aliases.ParameterVector,
    scan_resolution: int,
    dir_x: orqviz.aliases.DirectionVector,
    dir_y: orqviz.aliases.DirectionVector,
    cost_period: List[float],
    end_points: Tuple[float, float],
):
    scan2D_end_points_x, scan2D_end_points_y = get_scan_variables(
        cost_period, end_points
    )

    return orqviz.scans.perform_2D_scan(
        origin,
        cost_function,
        direction_x=dir_x,
        direction_y=dir_y,
        n_steps_x=scan_resolution,
        end_points_x=scan2D_end_points_x,
        end_points_y=scan2D_end_points_y,
    )


def perform_Fourier_scan(
    cost_function: orqviz.aliases.LossFunction,
    origin: orqviz.aliases.ParameterVector,
    scan_resolution: int,
    dir_x: orqviz.aliases.DirectionVector,
    dir_y: orqviz.aliases.DirectionVector,
    fourier_period: List[float],
    end_points: Tuple[float, float],
):
    scan2D_end_points_x, scan2D_end_points_y = get_fourier_plot_variables(
        fourier_period,
        end_points,
    )

    # GETTING DATA FOR FOURIER PLOTS
    scan2D_result_for_fourier = orqviz.scans.perform_2D_scan(
        origin,
        cost_function,
        direction_x=dir_x,
        direction_y=dir_y,
        n_steps_x=scan_resolution,
        end_points_x=scan2D_end_points_x,
        end_points_y=scan2D_end_points_y,
    )

    # Fourier landscape
    fourier_result = orqviz.fourier.perform_2D_fourier_transform(
        scan2D_result_for_fourier, scan2D_end_points_x, scan2D_end_points_y
    )

    return fourier_result


def _generate_metrics_dict(
    scan2D_result: orqviz.scans.Scan2DResult,
    fourier_result: orqviz.fourier.FourierResult,
    tv: float,
) -> Dict[str, float]:
    sparsity_from_norm = roughness_fourier_sparsity_using_norms(fourier_result)
    return {
        "tv": tv,
        "fourier density using norms": sparsity_from_norm,
    }


def load_data(
    directory_name: str, file_label: str
) -> Tuple[orqviz.scans.Scan2DResult, orqviz.fourier.FourierResult, Dict[str, float]]:
    scan2D_result = orqviz.io.load_viz_object(
        os.path.join(directory_name, file_label + "_scan2D")
    )
    fourier_result = orqviz.io.load_viz_object(
        os.path.join(directory_name, file_label + "_scan_fourier")
    )

    with open(os.path.join(directory_name, file_label + "_metrics.json"), "r") as f:
        metrics_dict = json.load(f)

    return scan2D_result, fourier_result, metrics_dict


def save_hamiltonians(
    hamiltonians_dict: Dict[str, PauliSum], timestamp_str: Optional[str] = None
):
    if timestamp_str is None:
        timestamp_str = generate_timestamp_str()
    directory_name = os.path.join(
        os.path.join(os.getcwd(), f"results/hamiltonians_{timestamp_str}")
    )
    # check if directory exists and create it if not

    if not os.path.exists(os.path.join(os.getcwd(), "results")):
        os.mkdir(os.path.join(os.getcwd(), "results"))
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    for label, hamiltonian in hamiltonians_dict.items():
        file_name = "{}.json".format(label)
        save_operator(hamiltonian, os.path.join(directory_name, file_name))


def load_hamiltonians(directory_name: str) -> Dict[str, PauliSum]:
    hamiltonians_dict: Dict[str, PauliSum] = {}
    for file_name in os.listdir(directory_name):
        label = file_name[0:-5]  # remove ".json"
        hamiltonians_dict[label] = load_operator(
            os.path.join(directory_name, file_name)
        )
    return hamiltonians_dict


def check_and_create_directory(directory_name: str, timestamp_str: str) -> str:
    # check if directory exists and create it if not
    if directory_name is None:
        if not os.path.exists(os.path.join(os.getcwd(), "results")):
            os.mkdir(os.path.join(os.getcwd(), "results"))
            print(f"Created directory {os.path.join(os.getcwd(), 'results')}")
    directory_name = os.path.join(os.getcwd(), f"results/data_{timestamp_str}")
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
        print(f"Created directory {directory_name}")

    return directory_name


def get_scan_variables(
    cost_period: List[float],
    default_end_points: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    freq_x = (2 * np.pi) / cost_period[0]
    freq_y = (2 * np.pi) / cost_period[1]
    scan2D_end_points_x = (
        default_end_points[0] / freq_x,
        default_end_points[1] / freq_x,
    )
    scan2D_end_points_y = (
        default_end_points[0] / freq_y,
        default_end_points[1] / freq_y,
    )

    return scan2D_end_points_x, scan2D_end_points_y


def get_fourier_plot_variables(
    fourier_period: List[float],
    default_end_points: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    freq_x = (2 * np.pi) / fourier_period[0]
    freq_y = (2 * np.pi) / fourier_period[1]
    scan2D_end_points_x = (
        default_end_points[0] / freq_x,
        default_end_points[1] / freq_x,
    )
    scan2D_end_points_y = (
        default_end_points[0] / freq_y,
        default_end_points[1] / freq_y,
    )

    return scan2D_end_points_x, scan2D_end_points_y


def save_scan_results(
    scan2D_result: orqviz.scans.Scan2DResult,
    fourier_result: orqviz.fourier.FourierResult,
    metrics_dict: Dict[str, float],
    directory_name: str,
    file_label: str,
) -> None:
    orqviz.io.save_viz_object(
        scan2D_result, os.path.join(directory_name, file_label + "_scan2D")
    )
    orqviz.io.save_viz_object(
        fourier_result, os.path.join(directory_name, file_label + "_scan_fourier")
    )
    with open(os.path.join(directory_name, file_label + "_metrics.json"), "w") as f:
        json.dump(metrics_dict, f)

    print("Saved data to directory: ", directory_name)
