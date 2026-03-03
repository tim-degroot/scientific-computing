import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from MonteCarlo import DLA
from GrayScott import GrayScott

PLOTS_DIR = "plots/"


def plot_grid(grid: np.ndarray, filename: str, plot: bool = False):
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap="binary", aspect="equal")
    plt.axis("off")
    plt.savefig(PLOTS_DIR + filename, bbox_inches="tight", pad_inches=0)
    plt.show() if plot else plt.close()


def plot_2x2_grids(grids: np.ndarray, labels: list, filename: str, plot: bool = False):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        ax.imshow(grids[i], cmap="binary", aspect="equal")

        ax.set_title(labels[i], fontsize=12)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(PLOTS_DIR + filename, bbox_inches="tight", pad_inches=0.1)
    plt.show() if plot else plt.close()


def plot_grid_overlap(grids: np.ndarray, filename: str, plot: bool = False):
    overlap_grid = np.sum(grids, axis=0)

    plt.figure(figsize=(7, 6))

    im = plt.imshow(overlap_grid, cmap="binary", aspect="equal")

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Number of Simulations")

    plt.axis("off")
    plt.savefig(PLOTS_DIR + filename, bbox_inches="tight", pad_inches=0.1)
    plt.show() if plot else plt.close()


def plot_2x2_overlap(
    grid_groups: np.ndarray, labels: list, filename: str, plot: bool = False
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    v_max = np.max([np.sum(g, axis=0) for g in grid_groups])

    for i in range(4):
        ax = axes[i]
        overlap = np.sum(grid_groups[i], axis=0)
        im = ax.imshow(overlap, cmap="binary", aspect="equal", vmin=0, vmax=v_max)
        ax.set_title(labels[i])

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        fraction=0.046,
        pad=0.04,
        label="Number of Simulations",
    )

    plt.savefig(PLOTS_DIR + filename, bbox_inches="tight")
    plt.show() if plot else plt.close()


def plot_4x4_grids(
    grids: np.ndarray, xlabels: list, ylabels: list, filename: str, plot: bool = False
):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), sharex=True, sharey=True)
    flat_grids = grids.reshape(16, grids.shape[-2], grids.shape[-1])
    flat_axes = axes.flatten()

    for i in range(16):
        ax = flat_axes[i]
        row = i // 4
        col = i % 4

        ax.imshow(flat_grids[i], cmap="binary", aspect="equal")

        if col == 0:
            ax.set_ylabel(ylabels[row])

        if row == 0:
            ax.set_xlabel(xlabels[col])
            ax.xaxis.set_label_position("top")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(PLOTS_DIR + filename, bbox_inches="tight", pad_inches=0.1)
    plt.show() if plot else plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Scientific Computing Set 2 simulations."
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help="Hide interactive plot windows and only save files.",
    )
    args = parser.parse_args()

    show_plots = not args.hide

    if "/" in PLOTS_DIR or "\\" in PLOTS_DIR:
        os.makedirs(os.path.dirname(PLOTS_DIR), exist_ok=True)

    size_x, size_y = (100, 100)
    seeds = [5, 8, 13, 21]
    seed_labels = [rf"seed = {seed:.0f}" for seed in seeds]
    ps_values = [0.25, 0.50, 0.75, 1.00]
    ps_labels = [rf"$p_s={ps_value:.2f}$" for ps_value in ps_values]

    print("Running Diffusion Limited Aggregation")

    print("Running Monte Carlo simulation of DLA")
    N_AGENTS = 500

    results = np.zeros((len(seeds), len(ps_values), size_x, size_y))

    for seed_index, seed in enumerate(seeds):
        for ps_index, ps_value in enumerate(ps_values):
            simulation = DLA(size_x=size_x, size_y=size_y, seed=seed, ps=ps_value)
            simulation.simulate_agents(N_AGENTS)
            results[seed_index, ps_index, :, :] = simulation.grid

    grids = results[:, 3, :, :]
    plot_2x2_grids(grids=grids, labels=seed_labels, filename=f"MC_DLA_grid")

    plot_4x4_grids(
        grids=results,
        xlabels=seed_labels,
        ylabels=ps_labels,
        filename="MC_DLA_matrix",
        plot=show_plots,
    )

    plot_2x2_overlap(
        grid_groups=results,
        labels=seed_labels,
        filename="MC_DLA_seed_overlap",
        plot=show_plots,
    )
    plot_2x2_overlap(
        grid_groups=np.transpose(results, (1, 0, 2, 3)),
        labels=ps_labels,
        filename="MC_DLA_ps_overlap",
        plot=show_plots,
    )
    print("Running The Gray-Scott model - A reaction-diffusion system")
