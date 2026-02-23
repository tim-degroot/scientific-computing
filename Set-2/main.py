import numpy as np
import matplotlib.pyplot as plt

from MonteCarlo import DLA

def plot_grid(grid: np.ndarray, filename: str):
    plt.figure(figsize=(5, 5))

    plt.imshow(grid, cmap='binary', aspect='equal')
    plt.axis('off')

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


for seed in [5, 8, 13, 21]:
    simulation = DLA(size_x=100, size_y=100, seed=seed)
    simulation.simulate_agents(100)

    plot_grid(grid=simulation.grid, filename=f"MonteCarloDLA_seed_{seed}")
    print(f"Finished simulation with seed = {seed}")