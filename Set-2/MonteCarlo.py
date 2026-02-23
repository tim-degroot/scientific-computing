import numpy as np
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import float64, int64

spec = [("grid", int64[:, :]), ("ps", float64)]


@jitclass(spec)
class DLA:
    def __init__(self, size_x: int, size_y: int, seed: int, ps: float = 1.0) -> None:
        self.grid = np.zeros((size_y, size_x), dtype=np.int64)
        self.ps = ps
        np.random.seed(seed)

        random_x = np.random.randint(0, self.grid.shape[1])
        self.grid[-1, random_x] = 1  # initialize a first point on the last row

    def simulate_agents(self, n: int):
        while np.count_nonzero(self.grid == 1) < n:
            self.simulate_agent()

    def simulate_agent(self) -> bool:
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
