import os
import sys
from typing import Dict

import matplotlib.ticker as tck
import orqviz

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.opt.optimizers.scipy_optimizer import ScipyOptimizer
from orquestra.quantum.operators import PauliSum, PauliTerm
from orqviz.plot_utils import get_colorbar_from_ax

from src.data_generation import generate_data, prepare_cost_function, save_hamiltonians
from src.plots import check_and_create_directory
from src.utils import calculate_plot_extents, generate_timestamp_str

color_bar_font_size = None
remove_constant = True
plt.rcParams["figure.dpi"] = 500
backend = QulacsSimulator()
scan_resolution = 201


def optimize_cost_function(cost_function, initial_params=None):
    is_finished = False
    exclude_borders = True
    counter = 0
    while not is_finished and exclude_borders:
        bound_limit = np.pi * 0.9
        bounds = [(-bound_limit, bound_limit), (-bound_limit, bound_limit)]
        if initial_params is None:
            initial_params = np.random.uniform(
                low=bounds[0][0], high=bounds[0][1], size=(2,)
            )
        else:
            exclude_borders = False
        optimizer = ScipyOptimizer(
            "L-BFGS-B",
            bounds=bounds,
        )
        results = optimizer.minimize(
            cost_function=cost_function, initial_params=initial_params
        )
        if np.isclose(np.abs(results.opt_params[0]), np.pi) or np.isclose(
            np.abs(results.opt_params[1]), np.pi
        ):
            is_finished = False
            initial_params = None
        else:
            is_finished = True
        counter += 1
        if counter > 5:
            print(f"Infinite loop mate: {counter}!")
    return results


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

    save_hamiltonians(hamiltonians_dict)

    cost_period = [2 * np.pi, 2 * np.pi]
    fourier_period = [2 * np.pi, 2 * np.pi]

    for file_label, hamiltonian in hamiltonians_dict.items():
        cost_function = prepare_cost_function(hamiltonian, backend)
        origin = np.array([0, 0])
        dir_x = np.array([1, 0])
        dir_y = np.array([0, 1])

        period, fourier_res_x, fourier_res_y = calculate_plot_extents(hamiltonian)
        fourier_res_x = 18
        fourier_res_y = 10
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

        plt.rcParams.update({"font.size": 15})

        check_and_create_directory(None, timestamp_str=timestamp)

        if remove_constant:
            my_values = np.copy(fourier_result.values)
            constant_term = my_values[0][0]
            my_values[0][0] = 0
            fourier_result = orqviz.fourier.FourierResult(
                my_values, fourier_result.end_points_x, fourier_result.end_points_y
            )
            constant_term_sign = -1 if np.mean(scan2D_result.values) > 0 else 1
            scan2D_result = orqviz.scans.clone_Scan2DResult_with_different_values(
                scan2D_result,
                scan2D_result.values + constant_term_sign * np.abs(constant_term),
            )

        fig, ax = plt.subplots(1, 1, constrained_layout=False)

        orqviz.scans.plot_2D_scan_result(scan2D_result, fig, ax)

        x, y = scan2D_result._get_coordinates_on_directions(in_units_of_direction=False)

        x_extent = np.pi
        y_extent = np.pi

        ax.xaxis.set_major_formatter(
            tck.FuncFormatter(
                lambda val, pos: "{:.2f}$\pi$".format(val / np.pi) if val != 0 else "0"
            )
        )
        ax.yaxis.set_major_formatter(
            tck.FuncFormatter(
                lambda val, pos: "{:.2f}$\pi$".format(val / np.pi) if val != 0 else "0"
            )
        )
        ax.tick_params(axis="both", labelsize=15)
        ax.xaxis.set_major_locator(tck.MultipleLocator(base=x_extent / 2))
        ax.yaxis.set_major_locator(tck.MultipleLocator(base=y_extent / 2))

        values = []
        n_samples = 100
        for _ in range(n_samples):
            results = optimize_cost_function(cost_function, initial_params=None)
            values.append(results.opt_value)
        #     ax.plot(results.opt_params[0], results.opt_params[1], "r*")

        ax.set_ylabel("$\\beta$")
        ax.set_xlabel("$\\gamma$")

        tv = metrics_dict.get("tv", 0)
        fourier_density = metrics_dict.get("fourier density using norms", 0)

        roughness_label = f"\n Total Variation: {'%.2f' % tv} \n Fourier Density: {'%.2f' % fourier_density} "
        fig.suptitle(roughness_label)

        plt.savefig(
            os.path.join(
                f"{os.getcwd()}/results/plots_{timestamp}",
                f"{file_label}_cost_landscape.png",
            )
        )

        plt.clf()
        plt.close()

        fig, ax = plt.subplots(1, 1, constrained_layout=False)

        orqviz.fourier.plot_2D_fourier_result(
            fourier_result=fourier_result,
            max_freq_x=fourier_res_x,
            max_freq_y=fourier_res_y,
            fig=fig,
            ax=ax,
        )

        x_axis_base = None
        if fourier_res_x <= 10:
            x_axis_base = 2
        elif fourier_res_x <= 20:
            x_axis_base = 4
        elif fourier_res_x <= 60:
            x_axis_base = 8
        elif fourier_res_x <= 100:
            x_axis_base = 16
        else:
            x_axis_base = 32

        ax.xaxis.set_major_locator(tck.MultipleLocator(base=x_axis_base))
        ax.yaxis.set_major_locator(tck.MultipleLocator(base=int(fourier_res_y / 2)))
        ax.tick_params(axis="both", labelsize=15)
        ax.set_xlabel("$f_{\\gamma}$")
        ax.set_ylabel("$f_{\\beta}$")

        if color_bar_font_size is not None:
            cbar1 = get_colorbar_from_ax(ax)
            cbar1.ax.tick_params(labelsize=color_bar_font_size)
            cbar2 = get_colorbar_from_ax(ax)
            cbar2.ax.tick_params(labelsize=color_bar_font_size)

        ax.set_box_aspect(1)

        plt.savefig(
            os.path.join(
                f"{os.getcwd()}/results/plots_{timestamp}",
                f"{file_label}_fourier.png",
            )
        )

        plt.clf()
        plt.close()

        fig, ax = plt.subplots(1, 1, constrained_layout=False)

        ax.hist(values, 100)
        ax.tick_params(axis="both", labelsize=15)
        ax.set_xlabel("cost")
        ax.set_ylabel("counts")
        ax.set_box_aspect(1)

        plt.savefig(
            os.path.join(
                f"{os.getcwd()}/results/plots_{timestamp}",
                f"{file_label}_histogram.png",
            )
        )

        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()
