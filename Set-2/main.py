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
        # ax.set_title(labels[i], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.close()


simulation = DLA(size_x=100, size_y=100, seed=13)
simulation.simulate_agents(500)
plot_grid(grid=simulation.grid, filename=f"MC_DLA_seed_13")

for seed in [8, 13, 21]:
    grids = []
    ps_values = [0.25, 0.50, 0.75, 1.00]
    for ps in ps_values:
        simulation = DLA(size_x=100, size_y=100, seed=seed, ps=ps)
        simulation.simulate_agents(500)

        grids.append(simulation.grid)
        
    print(f"Finished simulations with seed = {seed}")

    plot_2x2_comparison(
        grids=grids, labels=ps_values, filename=f"MC_DLA_seed_{seed}_comparison"
    )
