import numpy as np
import time

class D2Q9LBM:
    def __init__(self, nx, ny, tau, u_inlet):
        self.nx = nx
        self.ny = ny
        self.tau = max(0.51, tau)  # Ensure tau is above the stability threshold
        self.u_inlet = u_inlet
        
        # D2Q9 Lattice weights and directional velocities
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.c = np.array([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ])
        
        # Reverse indices for bounce-back boundaries
        self.opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # Initialize macroscopic variables
        self.rho = np.ones((nx, ny))
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        
        # Set inlet velocity profile (steady or parabolic)
        self.u[0, :] = u_inlet
        
        # Initialize populations to equilibrium
        self.f = self.equilibrium(self.rho, self.u, self.v)
        
        # Cylinder geometry mapping based on the 2.2m x 0.41m domain setup
        # Scaling: let's assume 1 grid unit = 0.01m
        self.obstacle = np.zeros((nx, ny), dtype=bool)
        cx, cy, r = 20, 20, 5 # Center at (0.2m, 0.2m) with 0.05m radius
        for i in range(nx):
            for j in range(ny):
                if (i - cx)**2 + (j - cy)**2 <= r**2:
                    self.obstacle[i, j] = True

    def equilibrium(self, rho, u, v):
        """Calculate the equilibrium distribution."""
        # Dynamically set the shape based on the input rho array
        feq = np.zeros((9,) + rho.shape)
        u2 = u**2 + v**2
        for i in range(9):
            cu = self.c[i, 0] * u + self.c[i, 1] * v
            cu = np.clip(cu, -1e3, 1e3)  # Clamp cu to avoid overflow
            feq[i] = self.w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
        return np.clip(feq, 0, 1e3)  # Clamp feq to avoid overflow

    def step(self):
        """Perform one LBM time step (Collision + Streaming)."""
        # 1. Macroscopic variables
        self.rho = np.sum(self.f, axis=0)
        self.rho = np.clip(self.rho, 1e-3, 1e3)  # Clamp rho to avoid invalid values
        self.u = np.sum(self.f * self.c[:, 0, np.newaxis, np.newaxis], axis=0) / self.rho
        self.v = np.sum(self.f * self.c[:, 1, np.newaxis, np.newaxis], axis=0) / self.rho
        
        # 2. Collision (BGK approximation)
        feq = self.equilibrium(self.rho, self.u, self.v)
        f_out = self.f - (self.f - feq) / self.tau
        
        # 3. Bounce-back boundary for the cylinder (No-slip U=V=0)
        for i in range(9):
            f_out[i, self.obstacle] = self.f[self.opposite[i], self.obstacle]
            
        # 4. Streaming
        for i in range(9):
            self.f[i] = np.roll(np.roll(f_out[i], self.c[i, 0], axis=0), self.c[i, 1], axis=1)
            
        # 5. Domain Boundaries (Simplified)
        # Top and bottom walls (No-slip U=V=0)
        for i in range(9):
            self.f[i, :, 0] = self.f[self.opposite[i], :, 0]
            self.f[i, :, -1] = self.f[self.opposite[i], :, -1]
            
        # Inlet (Zou-He velocity boundary approximation)
        self.u[0, :] = self.u_inlet
        self.v[0, :] = 0
        self.rho[0, :] = 1.0 # Simple approximation
        self.f[:, 0, :] = self.equilibrium(self.rho[0, :], self.u[0, :], self.v[0, :])
        
        # Outlet (Open boundary)
        self.f[:, -1, :] = self.f[:, -2, :]