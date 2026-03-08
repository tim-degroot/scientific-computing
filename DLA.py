import numpy as np
import math
from numba.experimental import jitclass
from numba import int64, float64

spec = [
    ("size_x", int64),
    ("size_y", int64),
    ("grid", int64[:, :]),
    ("eta", float64),
    ("c_history", float64[:, :, :]),
]


@jitclass(spec)
class DLA:
    def __init__(self, size_x: int, size_y: int, eta: float, omega: float, seed: int):
        self.size_x = size_x
        self.size_y = size_y
        self.grid = np.zeros((size_y+2, size_x), dtype=np.int64)
        self.eta = eta
        self.omega = omega
        np.random.seed(seed)

        midpoint = int(math.ceil(size_x / 2))
        self.grid[-2, midpoint] = 1

    def SOR_Iteration(self, epsilon=1e-5, max_iter=10000):
        """
        Find steady state using SOR method
        """
        c = np.zeros((self.size_y+2, self.size_x), dtype=np.float64)
        # includes a emission and absorbtion row at the top and bottom
        c[0, :] = 1.0
        c_new = np.copy(c)

        it = 0
        while it < max_iter:
            for j in range(1, self.size_y+1):
                for i in range(self.size_x):
                    if self.grid[j, i] == 1:
                        c[j, i] = 0.0

            c_new[0, :] = 1.0
            c_new[-1, :] = 0.0

            for j in range(1, self.size_y+1):
                for i in range(self.size_x):
                    if self.grid[j, i] == 0:
                        # j_down = min(j + 1, self.size_y - 1)
                        c_new[j, i] = (self.omega / 4.0) * (
                            c[j, (i + 1) % self.size_x]
                            + c_new[j, (i - 1) % self.size_x]
                            + c[j+1, i]
                            + c_new[j - 1, i]
                        ) + (1.0 - self.omega) * c[j, i]
            it += 1
            delta = np.absolute(c_new - c)
            if np.max(delta) < epsilon:
                return np.around(c_new, 2), it
            c = np.copy(c_new)
        return np.around(c_new, 2), it

    def neighbour_in_cluster(self, x: int, y: int) -> bool:
        width = self.size_x
        height = self.size_y + 2

        if self.grid[y, (x - 1) % width] == 1:
            return True
        if self.grid[y, (x + 1) % width] == 1:
            return True
        if y - 1 >= 0 and self.grid[y - 1, x] == 1:
            return True
        if y + 1 < height and self.grid[y + 1, x] == 1:
            return True

        return False

    def find_neighbors(self):
        neighbors = [(0, 0)]
        neighbors.pop()

        for j in range(1, self.size_y+1):
            for i in range(self.size_x):
                if self.grid[j, i] == 0 and self.neighbour_in_cluster(i, j):
                    neighbors.append((i, j))
        return neighbors

    def one_growth_step(self):
        neighbors = self.find_neighbors()
        c, it = self.SOR_Iteration()

        if self.eta == 0.0:
            random_index = np.random.randint(len(neighbors))
        else:
            probs = np.zeros(len(neighbors), dtype=np.float64)
            for idx in range(len(neighbors)):
                x, y = neighbors[idx]
                # val = max(c[y, x], 1e-15)
                val = c[y, x]
                probs[idx] = self.eta * np.log(val)

            probs -= np.max(probs)
            probs = np.exp(probs)
            probs /= np.sum(probs)


            r = np.random.random()
            acc = 0.0
            random_index = 0
            for i in range(len(probs)):
                acc += probs[i]
                if r <= acc:
                    random_index = i
                    break

        new_object = neighbors[random_index]
        self.grid[new_object[1], new_object[0]] = 1

        return c

    def simulate_growth_model(self, steps: int):
        c_history = np.zeros((self.size_y+2, self.size_x, steps+1), dtype=np.float64)
        for t in range(steps):
            # if t % 50 == 0:
            #     print("Step", t)
            c = self.one_growth_step()
            c_history[:, :, t] = c
        
        c_final, _ = self.SOR_Iteration()
        c_history[:, :, -1] = c_final
        
        self.grid = self.grid[1:-1, :]
        return c_history
