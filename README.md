# Connecting the Hamiltonian structure to the QAOA energy and Fourier landscape structure by [Michał Stęchły](@mstechly), [Laura Gao](@laurgao), [Matthew S. Rudolph](@MSRudolph), [Enrico Fontana](@e-font) and [Boniface Yogendran](@Boniface316).

This repository contains the code required to reproduce the results presented in the paper titled "Connecting the Hamiltonian structure to the QAOA energy and Fourier landscape structure." The paper is available on arXiv: [arXiv:2305.13594](https://arxiv.org/abs/2305.13594).

## Running the experiment

All the scripts used to produce the plots in the paper are located in the `paper_plots` folder. The scripts are organized into folders corresponding to the figures in the paper. Each folder contains a `main.py` script that runs the experiment and produces the plots. The `main.py` script can be run from the command line. For example, to run the experiment for Figure 1, run the following command:

`bash python paper_plots/figure_1/main.py`

The outcomes of the script are saved in the `results` folder. For each script, it produces `hamiltionian`, `data`, and `plots` sub-folders. These sub-folders have timestamp concatenated to their names i.e `data_2023_06_19_22_25_38`. The `hamiltonian` folder contains the Hamiltonian in form of `json` file. Each Hamiltonian produces 3 files in the `data` folder. They correspond to metrics, 2D scan and Fourier outcome. The `plots` folder contains the plots produced by the experiment in the form of `.png` files.

We encourage you to explore the code and modify it to suit your needs. Experiment with various Hamiltonians and see how the cost landscape evolves.

If you need more materials to understand how to use `Orqviz` package to visualize Fourier landscapes, please refer to the [Orqviz Fourier transform notebook](https://github.com/zapatacomputing/orqviz/blob/main/docs/examples/fourier_transform.ipynb).

We hope you find this repository useful and that it helps in understanding the Hamiltonian structure in relation to the QAOA energy and Fourier landscape structure.
