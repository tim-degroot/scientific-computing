import math
import numpy as np
from numba.experimental import jitclass
from numba import float64, int64

spec = [("grid", int64[:, :]), ("ps", float64)]


@jitclass(spec)
class DLA:
    """
    Diffusion-Limited Aggregation (DLA) simulation.
    
    This class models the formation of clusters by simulating random-walking 
    particles that stick together upon contact. It uses periodic boundary 
    conditions along the x-axis and stops particles that move beyond the y-axis bounds.
    
    Attributes:
        grid (np.ndarray): A 2D array representing the simulation space where 
            0 is empty space and 1 is a part of the cluster.
        ps (float): The sticking probability; the chance a particle attaches 
            to the cluster when touching it.
    """

    def __init__(self, size_x: int, size_y: int, seed: int, ps: float = 1.0) -> None:
        """
        Initializes the DLA simulation grid and places the initial seed particle.
        
        Args:
            size_x (int): The width of the simulation grid.
            size_y (int): The height of the simulation grid.
            seed (int): The random seed for reproducibility.
            ps (float, optional): The sticking probability. Defaults to 1.0.
        """
        self.grid = np.zeros((size_y, size_x), dtype=np.int64)
        self.ps = ps
        np.random.seed(seed)

        midpoint = math.ceil(size_x / 2)
        self.grid[-1, midpoint] = 1

    def simulate_agents(self, n: int) -> None:
        """
        Simulates particles continuously until a target cluster size is reached.
        
        Args:
            n (int): The target number of particles to have in the cluster.
        """
        while np.count_nonzero(self.grid == 1) < n:
            self.simulate_agent()

    def simulate_agent(self) -> bool:
        """
        Simulates the random walk of a single agent starting from the top row.
        
        The agent walks randomly (up, down, left, right) until it either 
        touches the cluster and sticks (based on the sticking probability `ps`), 
        or walks off the top/bottom boundaries of the grid.
        
        Returns:
            bool: True if the agent successfully attached to the cluster, 
            False if it exited the grid boundaries.
        """
        random_x = np.random.randint(0, self.grid.shape[1])
        y, x = 0, random_x

        if self.neighbour_in_cluster(x, y):
            if np.random.rand() < self.ps:
                self.grid[y, x] = 1
                return True

        while True:
            direction = np.random.randint(0, 4)
            new_y, new_x = y, x

            if direction == 0:
                new_y -= 1  # "u"
            elif direction == 1:
                new_y += 1  #  "d"
            elif direction == 2:
                new_x = (new_x - 1) % self.grid.shape[1]  # "l"
            else:
                new_x = (new_x + 1) % self.grid.shape[1]  # "r"

            if new_y < 0 or new_y >= self.grid.shape[0]:
                return False

            if self.grid[new_y, new_x] == 1:
                continue

            y, x = new_y, new_x

            if self.neighbour_in_cluster(x, y):
                if np.random.rand() < self.ps:
                    self.grid[y, x] = 1
                    return True

    def neighbour_in_cluster(self, x: int, y: int) -> bool:
        """
        Checks if the given coordinates are adjacent to an existing cluster cell.
        
        Uses a von Neumann neighborhood (up, down, left, right) and evaluates 
        the x-axis with periodic boundary conditions (wrapping around the edges).
        
        Args:
            x (int): The x-coordinate of the agent.
            y (int): The y-coordinate of the agent.
            
        Returns:
            bool: True if at least one adjacent cell contains a cluster particle, 
            False otherwise.
        """
        width = self.grid.shape[1]
        height = self.grid.shape[0]

        if self.grid[y, (x - 1) % width] == 1:
            return True
        if self.grid[y, (x + 1) % width] == 1:
            return True
        if y - 1 >= 0 and self.grid[y - 1, x] == 1:
            return True
        if y + 1 < height and self.grid[y + 1, x] == 1:
            return True

        return False
