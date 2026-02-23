import numpy as np
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import int64

spec = [
    ('grid', int64[:, :]), 
]

@jitclass(spec)
class DLA:
    def __init__(self, size_x: int, size_y: int,  seed: int) -> None:
        self.grid = np.zeros((size_y, size_x), dtype=np.int64)
        np.random.seed(seed)

        random_x = np.random.randint(0, self.grid.shape[1])
        self.grid[-1, random_x] = 1  # initialize a first point on the last row

    def simulate_agents(self, n: int):
        while np.count_nonzero(self.grid == 1) <= n:
            self.simulate_agent()

    def simulate_agent(self) -> bool:
        random_x = np.random.randint(0, self.grid.shape[1])
        y, x = 0, random_x

        while True:
            if y < 0 or y >= self.grid.shape[0]:
                return False

            if self.neighbour_in_cluster(x, y):
                self.grid[y, x] = 1
                return True

            direction = np.random.randint(0, 4)
            if direction == 0:   
                y -= 1 # "u"
            elif direction == 1: 
                y += 1 #  "d"
            elif direction == 2:
                x = (x - 1) % self.grid.shape[1] # "l"
            else:                
                x = (x + 1) % self.grid.shape[1] # "r"

    def neighbour_in_cluster(self, x: int, y: int) -> bool:
        neighbours = [
            self.grid[y, (x - 1) % self.grid.shape[1]],
            self.grid[y, (x + 1) % self.grid.shape[1]],
        ]

        if y - 1 >= 0:
            neighbours.append(self.grid[y - 1, x])

        if y + 1 < self.grid.shape[0]:
            neighbours.append(self.grid[y + 1, x])

        for n in neighbours:
            if n == 1:
                return True
        return False