import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo import DLA


def plot_grid(grid: np.ndarray, filename: str):
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap="binary", aspect="equal")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_2x2_comparison(grids: list, labels: list, filename: str):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        ax.imshow(grids[i], cmap="binary", aspect="equal")
        ax.set_title(rf"$p_s$ = {labels[i]:.0%}", fontsize=12, pad=10)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.2)
    plt.close()


if __name__ == "__main__":
    print("Running Diffusion Limited Aggregation")

    print("Running Monte Carlo simulation of DLA")
    simulation = DLA(size_x=100, size_y=100, seed=8)
    simulation.simulate_agents(500)
    plot_grid(grid=simulation.grid, filename=f"MC_DLA_seed_8")

    seeds = [8, 13, 21]
    ps_values = [0.25, 0.50, 0.75, 1.00]

    for seed in seeds:
        grids = []

        for ps in ps_values:
            simulation = DLA(size_x=100, size_y=100, seed=seed, ps=ps)
            simulation.simulate_agents(500)
            grids.append(simulation.grid)

        print(f"Finished simulations with seed = {seed}")

        plot_2x2_comparison(
            grids=grids, labels=ps_values, filename=f"MC_DLA_seed_{seed}_comparison"
        )

    print("Running The Gray-Scott model - A reaction-diffusion system")
