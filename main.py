import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from MonteCarlo import MCDLA
from GrayScott import GrayScott
from DLA import DLA

PLOTS_DIR = "plots/"
DATA_DIR = "data/"
PLOTS_FORMAT = ".pdf"


def plot_grid(grid: np.ndarray, filename: str, plot: bool = False):
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap="binary", aspect="equal")
    plt.axis("off")
    plt.savefig(PLOTS_DIR + filename + PLOTS_FORMAT, bbox_inches="tight", pad_inches=0)
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
    plt.savefig(
        PLOTS_DIR + filename + PLOTS_FORMAT, bbox_inches="tight", pad_inches=0.1
    )
    plt.show() if plot else plt.close()


def plot_grid_overlap(grids: np.ndarray, filename: str, plot: bool = False):
    overlap_grid = np.sum(grids, axis=0)

    plt.figure(figsize=(7, 6))

    im = plt.imshow(overlap_grid, cmap="binary", aspect="equal")

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Number of Simulations")

    plt.axis("off")
    plt.savefig(
        PLOTS_DIR + filename + PLOTS_FORMAT, bbox_inches="tight", pad_inches=0.1
    )
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

    plt.savefig(PLOTS_DIR + filename + PLOTS_FORMAT, bbox_inches="tight")
    plt.show() if plot else plt.close()


def plot_4x4_grids(
    grids: np.ndarray, xlabels: list, ylabels: list, filename: str, plot: bool = False,
):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), sharex=True, sharey=True)
    flat_axes = axes.flatten()

    for i in range(16):
        ax = flat_axes[i]
        row = i // 4
        col = i % 4

        ax.imshow(grids[col, row], cmap="binary", aspect="equal")

        if col == 0:
            ax.set_ylabel(ylabels[row])

        if row == 0:
            ax.set_xlabel(xlabels[col])
            ax.xaxis.set_label_position("top")

        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        PLOTS_DIR + filename + PLOTS_FORMAT, bbox_inches="tight", pad_inches=0.1
    )
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
    parser.add_argument("--load", action="store_true", help="Load previous simulations")
    args = parser.parse_args()

    show_plots = not args.hide
    load_data = args.load

    if "/" in PLOTS_DIR or "\\" in PLOTS_DIR:
        os.makedirs(os.path.dirname(PLOTS_DIR), exist_ok=True)

    if "/" in DATA_DIR or "\\" in DATA_DIR:
        os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)

    STEPS = 500
    size_x, size_y = (100, 100)
    seeds = [5, 8, 13, 21]
    eta_values = np.arange(0.0, 2.0, step=0.5)
    eta_labels = [rf"$\eta={eta_value:.2f}$" for eta_value in eta_values]

    seed_labels = [rf"seed = {seed:.0f}" for seed in seeds]
    ps_values = [0.25, 0.50, 0.75, 1.00]
    ps_labels = [rf"$p_s={ps_value:.2f}$" for ps_value in ps_values]

    print("Running Diffusion Limited Aggregation")
    results = np.zeros((len(seeds), len(eta_values), size_x, size_y))

    if load_data:
        results = np.load(f"{DATA_DIR}DLA_results.npy")
    else:
        total_dla_iters = len(seeds) * len(eta_values)
        with tqdm(total=total_dla_iters, desc="Running Simulations") as pbar:
            for seed_index, seed in enumerate(seeds):
                for eta_index, eta_value in enumerate(eta_values):
                    model = DLA(size_x=size_x, size_y=size_y, eta=eta_value, omega=1.9, seed=seed)
                    model.simulate_growth_model(500)

                    results[seed_index, eta_index, :, :] = model.grid
                    pbar.update(1)

    np.save(f"{DATA_DIR}DLA_results", results)

    grids = results[:, 2, :, :]
    plot_2x2_grids(grids=grids, labels=seed_labels, filename=f"DLA_grid", plot=True)

    plot_4x4_grids(
        grids=results,
        xlabels=seed_labels,
        ylabels=eta_labels,
        filename="DLA_matrix",
        plot=show_plots,
    )

    plot_2x2_overlap(
        grid_groups=np.transpose(results, (1, 0, 2, 3)),
        labels=eta_labels,
        filename="DLA_seed_overlap",
        plot=show_plots,
    )

    # exit()

    print("Running Monte Carlo simulation of DLA")
    results = np.zeros((len(seeds), len(ps_values), size_x, size_y))

    if load_data:
        results = np.load(f"{DATA_DIR}MC_DLA_results.npy")
    else:
        total_dla_iters = len(seeds) * len(ps_values)
        with tqdm(total=total_dla_iters, desc="Running Simulations") as pbar:
            for seed_index, seed in enumerate(seeds):
                for ps_index, ps_value in enumerate(ps_values):
                    simulation = MCDLA(
                        size_x=size_x, size_y=size_y, seed=seed, ps=ps_value
                    )
                    simulation.simulate_agents(STEPS)
                    results[seed_index, ps_index, :, :] = simulation.grid
                    pbar.update(1)

    np.save(f"{DATA_DIR}MC_DLA_results", results)

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

    # Intervals
    N = 100
    n_timesteps = 40000
    noise = 0.01  # +-1%

    # Get inital conditions for u and v
    u_init, v_init = GrayScott.initial_conditions(N, 0.5, 0.25, 10, 0.01)

    # Define parameters to plot
    args_set = [
        {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, "feed": 0.035, "kill": 0.06},
        {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, "feed": 0.05, "kill": 0.065},
        {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, "feed": 0.015, "kill": 0.044},
        {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, "feed": 0.04, "kill": 0.065},
    ]

    savefile = PLOTS_DIR + "GrayScott" + PLOTS_FORMAT
    GrayScott.plot_argsets(
        u_init, v_init, n_timesteps, args_set, savefile, show=show_plots
    )
