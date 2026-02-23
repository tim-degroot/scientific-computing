import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from MonteCarlo import DLA


def plot_grid(grid: np.ndarray, filename: str):
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap="binary", aspect="equal")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()

for seed in [8, 13, 21]:
    simulation = DLA(size_x=100, size_y=100, seed=seed, ps=1)
    simulation.simulate_agents(100)

    plot_grid(grid=simulation.grid, filename=f"MonteCarloDLA_seed_{seed}")
    print(f"Finished simulation with seed = {seed}")


for ps in np.linspace(0.2, 0.8, 4):
    simulation = DLA(size_x=100, size_y=100, seed=13, ps=ps)
    simulation.simulate_agents(100)

    plot_grid(grid=simulation.grid, filename=f"MonteCarloDLA_ps_{ps:.0%}")
    print(f"Finished simulation with ps = {ps:.0%}")

