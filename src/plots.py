import math
import os
import warnings
from typing import Dict, List, Optional, Union

import boto3
import matplotlib.ticker as tck
import numpy as np
import orqviz
from matplotlib import pyplot as plt
from orqviz.fourier import _check_and_create_fig_ax
from orqviz.plot_utils import get_colorbar_from_ax

from .utils import generate_timestamp_str

plt.rcParams["figure.dpi"] = 500


def custom_fourier_plot(
    fourier_result: orqviz.fourier.FourierResult, fig, ax, fourier_res_x, fourier_res_y
):
    breakpoint()

    raw_values = fourier_result.values
    size = raw_values.shape[0]

    half = int(size / 2)
    min_x = half
    max_x = int(half + fourier_res_x)
    min_y = int(half - fourier_res_y) - 1
    max_y = int(half + fourier_res_y) + 1

    result = fourier_result
    plottable_result = np.abs(result.values)

    # Remove constant term
    plottable_result[half][half] = 0
    truncated_result = plottable_result[min_y:max_y, min_x:max_x]

    n_x = truncated_result.shape[1]
    n_y = truncated_result.shape[0]

    x_axis = np.arange(0, n_x)
    y_axis = np.arange(-n_y / 2, n_y / 2)
    # you want the extra for the positive side
    XX, YY = np.meshgrid(x_axis, y_axis)

    default_plot_kwargs = {"shading": "auto"}

    fig, ax = _check_and_create_fig_ax(fig=fig, ax=ax)
    mesh_plot = ax.pcolormesh(
        XX, YY, truncated_result, **default_plot_kwargs, rasterized=True
    )

    fig.colorbar(mesh_plot, ax=ax)
    ax.set_xlabel("Scan Direction x")
    ax.set_ylabel("Scan Direction y")
    ax.set_ylim(-fourier_res_y, fourier_res_y)


def create_plot(
    scan2D_result: orqviz.scans.Scan2DResult,
    fourier_result: orqviz.fourier.FourierResult,
    metrics_dict: Dict[str, float],
    label: str,
    fourier_res_x: int,
    fourier_res_y: int,
    directory_name: Optional[str] = None,
    timestamp_str: Optional[str] = None,
    unit: str = "tau",
    remove_constant: bool = False,
    include_all_metrics: bool = True,
    color_bar_font_size: Optional[int] = None,
    plot_title: str = "",
    fix_x_y_extents: bool = False,
    font_size=15,
):
    """
    fourier_res: maximum frequency to keep on the plot (inclusive), calculated by
        summing up the coefficients of the operator
    """
    plt.rcParams.update({"font.size": font_size})

    if timestamp_str is None:
        timestamp_str = generate_timestamp_str()

    if directory_name is None:
        directory_name = os.path.join(os.getcwd(), f"results/plots_{timestamp_str}")
        if not os.path.exists(os.path.join(os.getcwd(), "results")):
            os.mkdir(os.path.join(os.getcwd(), "results"))

    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    if remove_constant:
        scan2D_result, fourier_result = remove_constant_term(
            fourier_result, scan2D_result
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    orqviz.scans.plot_2D_scan_result(scan2D_result, fig, ax1)

    x_extent, y_extent = get_x_y_extents(scan2D_result, fix_x_y_extents)

    if unit == "tau":
        tau = np.pi * 2
        ax1.xaxis.set_major_formatter(
            tck.FuncFormatter(
                lambda val, pos: "{:.2f}$\\tau$".format(val / tau) if val != 0 else "0"
            )
        )
        ax1.yaxis.set_major_formatter(
            tck.FuncFormatter(
                lambda val, pos: "{:.2f}$\\tau$".format(val / tau) if val != 0 else "0"
            )
        )

        ax1.xaxis.set_major_locator(tck.MultipleLocator(base=x_extent / 4))
        ax1.yaxis.set_major_locator(tck.MultipleLocator(base=y_extent / 4))
    elif unit == "pi":
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
    else:
        raise ValueError("unit must be either 'tau' or 'pi'")
    ax1.tick_params(axis="both", labelsize=15)
    ax1.xaxis.set_major_locator(tck.MultipleLocator(base=x_extent / 4))
    ax1.yaxis.set_major_locator(tck.MultipleLocator(base=y_extent / 4))
    ax1.set_xlabel("$\\gamma$")
    ax1.set_ylabel("$\\beta$")

    custom_fourier_plot(
        fourier_result=fourier_result,
        fig=fig,
        ax=ax2,
        fourier_res_x=fourier_res_x,
        fourier_res_y=fourier_res_y,
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

    ax2.xaxis.set_major_locator(tck.MultipleLocator(base=x_axis_base))
    ax2.yaxis.set_major_locator(tck.MultipleLocator(base=int(fourier_res_y / 2)))
    # ax2.set_yticks(np.arange(-fourier_res_y, fourier_res_y, int(fourier_res_y / 4)))
    ax2.tick_params(axis="both", labelsize=15)
    ax2.set_xlabel("$f_{\\gamma}$")
    ax2.set_ylabel("$f_{\\beta}$")

    if color_bar_font_size is not None:
        cbar1 = get_colorbar_from_ax(ax1)
        cbar1.ax.tick_params(labelsize=color_bar_font_size)
        cbar2 = get_colorbar_from_ax(ax2)
        cbar2.ax.tick_params(labelsize=color_bar_font_size)

    # Round to 2 decimal places

    title_of_the_plot = get_plot_title(include_all_metrics, metrics_dict, plot_title)

    fig.suptitle(title_of_the_plot)
    fig.tight_layout()

    plt.savefig(os.path.join(directory_name, f"{label}.png"))
    print(f"Saved plot to {os.path.join(directory_name, f'{label}.png')}")
    # plt.savefig(f"{label}.png")

    plt.clf()
    plt.close()


def remove_constant_term(
    fourier_result: orqviz.fourier.FourierResult,
    scan2D_result: orqviz.scans.Scan2DResult,
):
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

    return scan2D_result, fourier_result


def get_x_y_extents(scan2D_result, fix_x_y_extents):
    min_x = scan2D_result.params_grid[0][0][0]
    max_x = scan2D_result.params_grid[0][-1][0]
    min_y = scan2D_result.params_grid[0][0][1]
    max_y = scan2D_result.params_grid[-1][0][1]

    if fix_x_y_extents:

        x_extent = np.pi
        y_extent = np.pi

    else:
        x_extent = max_x - min_x
        y_extent = max_y - min_y

    return x_extent, y_extent


def get_plot_title(
    include_all_metrics: bool, metrics_dict: Dict[str, float], plot_title: str
):
    if include_all_metrics:
        roughness_label = " \n ".join(
            [f"Roughness index [{k}]: {'%.2f' % v}" for k, v in metrics_dict.items()]
        )
    else:
        tv = metrics_dict.get("tv", 0)
        fourier_density = metrics_dict.get("fourier density using norms", 0)

        roughness_label = f"\n Total variation: {'%.2f' % tv} \n Fourier density: {'%.2f' % fourier_density} "
    return plot_title + roughness_label
