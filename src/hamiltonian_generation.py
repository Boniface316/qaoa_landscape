import random
from itertools import combinations
from typing import Tuple

import networkx as nx
from orquestra.opt.problems.maxcut import MaxCut
from orquestra.quantum.operators import PauliSum, PauliTerm

### General generators


def single_order_N(n_qubits: int, order: int) -> PauliSum:
    Pauli_sum = PauliSum()
    nodes = [i for i in range(n_qubits)]

    pauli_term_in_order = []
    terms = combinations(nodes, order)
    for graph_nodes in terms:
        pauliterms = PauliTerm("I0", 1)
        for graph_node in graph_nodes:
            pauliterms *= PauliTerm("Z" + f"{graph_node}")
        pauli_term_in_order.append(pauliterms)
    Pauli_sum += PauliSum(pauli_term_in_order)

    return Pauli_sum


def all_linear_terms(n_qubits: int) -> PauliSum:
    return single_order_N(n_qubits, 1)


def all_quadratic_terms(n_qubits: int) -> PauliSum:
    return single_order_N(n_qubits, 2)


def all_cubic_terms(n_qubits: int) -> PauliSum:
    return single_order_N(n_qubits, 3)


def all_possible_terms(n_qubits: int) -> PauliSum:
    Pauli_linear_terms = all_linear_terms(n_qubits)
    Pauli_quadratic_terms = all_quadratic_terms(n_qubits)
    Pauli_cubic_terms = all_cubic_terms(n_qubits)

    return Pauli_linear_terms + Pauli_quadratic_terms + Pauli_cubic_terms


def all_terms_of_order_N(n_qubits: int, order: int) -> PauliSum:
    Pauli_sum = PauliSum()
    for order_term in range(order + 1):
        Pauli_sum += single_order_N(n_qubits, order_term)
    Pauli_sum.terms.pop(0)

    return Pauli_sum


def assign_random_weights(
    operator: PauliSum, use_integers: bool = True, range: Tuple[int, int] = [-10, 10]
) -> PauliSum:

    for i, term in enumerate(operator.terms):
        if use_integers:
            term_weight = random.randint(range[0], range[1])
        else:
            term_weight = random.uniform(range[0], range[1])

        operator.terms[i] = term * term_weight

    return operator


def assign_weights_with_common_denominator(
    operator: PauliSum, denominator: int, max_multiplier: int
) -> PauliSum:

    for i, term in enumerate(operator.terms):
        term_weight = random.randrange(
            denominator, max_multiplier + denominator, denominator
        )
        operator.terms[i] = term * term_weight

    return operator


def assign_weight_for_term(
    operator: PauliSum, term: PauliTerm, weight: float
) -> PauliSum:
    if term.circuit in operator.circuits:
        location_of_operator = operator.circuits.index(term.circuit)
        operator.terms[location_of_operator].coefficient = weight
    else:
        raise ValueError("term is not found in the operator")

    return operator


# provide a term and and the weight will be assigned to it


def assign_percentage_weight_for_term(
    operator: PauliSum, term: PauliTerm, weight: float
) -> PauliSum:
    number_of_terms = len(operator.terms)
    sum_of_coefficients = sum(
        operator.terms[i].coefficient.real for i in range(number_of_terms)
    )

    if term.circuit in operator.circuits:
        location_of_operator = operator.circuits.index(term.circuit)
        operator_weight = operator.terms[location_of_operator].coefficient
    else:
        raise ValueError("term is not found in the operator")

    if round(operator_weight.real / sum_of_coefficients) == weight:
        return operator

    else:
        remainder = sum_of_coefficients - operator_weight
        new_weight = weight * remainder / (1 - weight)

        operator.terms[location_of_operator].coefficient = new_weight

    return operator


### Special cases


def star_quadratic_terms(n_qubits: int) -> PauliSum:
    graph = nx.star_graph(n_qubits)
    pauli_term = []
    Pauli_sum = PauliSum()
    for node1, node2 in graph.edges():
        pauli_term.append(PauliTerm("Z" + f"{node1}") * PauliTerm("Z" + f"{node2}"))
    Pauli_sum += PauliSum(pauli_term)
    return Pauli_sum


def linear_quadratic_terms(n_qubits: int) -> PauliSum:
    graph = nx.path_graph(n_qubits)
    Pauli_sum = PauliSum()
    pauli_term = []
    for node1, node2 in graph.edges():
        pauli_term.append(PauliTerm("Z" + f"{node1}") * PauliTerm("Z" + f"{node2}"))
    Pauli_sum += PauliSum(pauli_term)
    return Pauli_sum


def linear_cubic_terms(n_qubits: int) -> PauliSum:
    Pauli_sum = PauliSum()
    pauli_term = []
    for node in range(n_qubits - 1):
        pauli_term.append(
            PauliTerm("Z" + f"{node}")
            * PauliTerm("Z" + f"{node+1}")
            * PauliTerm("Z" + f"{node+2}")
        )
    Pauli_sum += PauliSum(pauli_term)
    return Pauli_sum


def linear_quatric_terms(n_qubits: int) -> PauliSum:
    Pauli_sum = PauliSum()
    pauli_term = []
    for node in range(n_qubits - 2):
        pauli_term.append(
            PauliTerm("Z" + f"{node}")
            * PauliTerm("Z" + f"{node+1}")
            * PauliTerm("Z" + f"{node+2}")
            * PauliTerm("Z" + f"{node+3}")
        )
    Pauli_sum += PauliSum(pauli_term)
    return Pauli_sum


def erdos_renyi_Pauli_sum(num_qubits: int, probability: float, seed=int):
    graph = nx.erdos_renyi_graph(n=num_qubits, p=probability, seed=seed)
    return MaxCut().get_hamiltonian(graph)
