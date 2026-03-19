import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class D2Q9LBM:
    def __init__(self, resolution, tau, u_inlet):
        # Fixed domain dimensions from the image
        domain_width = 2.2  # meters
        domain_height = 0.41  # meters

        self.resolution = resolution  # STORE RESOLUTION FOR PLOTTING

        # Calculate grid size based on resolution
        self.nx = int(domain_width / resolution)
        self.ny = int(domain_height / resolution)
        self.tau = max(0.51, tau)  # Ensure tau is above the stability threshold
        self.u_inlet = u_inlet

        # D2Q9 Lattice weights and directional velocities
        self.w = np.array(
            [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
        )
        self.c = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ]
        )

        # Reverse indices for bounce-back boundaries
        self.opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # Initialize macroscopic variables
        self.rho = np.ones((self.nx, self.ny))
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))

        # Set inlet velocity profile
        self.u[0, :] = u_inlet

        # Initialize populations to equilibrium
        self.f = self.equilibrium(self.rho, self.u, self.v)

        # Cylinder geometry mapping based on the 2.2m x 0.41m domain setup
        self.obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        cx, cy, r = (
            int(0.15 / resolution),
            int(0.21 / resolution),
            int(0.05 / resolution),
        )
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - cx) ** 2 + (j - cy) ** 2 <= r**2:
                    self.obstacle[i, j] = True

    def equilibrium(self, rho, u, v):
        """Calculate the equilibrium distribution."""
        feq = np.zeros((9,) + rho.shape)
        u2 = u**2 + v**2
        for i in range(9):
            cu = self.c[i, 0] * u + self.c[i, 1] * v
            cu = np.clip(cu, -1e3, 1e3)  # Clamp cu to avoid overflow
            feq[i] = self.w[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return np.clip(feq, 0, 1e3)  # Clamp feq to avoid overflow

    def step(self):
        """Perform one LBM time step (Collision + Streaming)."""
        # 1. Macroscopic variables
        self.rho = np.sum(self.f, axis=0)
        self.rho = np.clip(self.rho, 1e-3, 1e3)  # Clamp rho to avoid invalid values
        self.u = (
            np.sum(self.f * self.c[:, 0, np.newaxis, np.newaxis], axis=0) / self.rho
        )
        self.v = (
            np.sum(self.f * self.c[:, 1, np.newaxis, np.newaxis], axis=0) / self.rho
        )

        # 2. Collision (BGK approximation)
        feq = self.equilibrium(self.rho, self.u, self.v)
        f_out = self.f - (self.f - feq) / self.tau

        # 3. Bounce-back boundary for the cylinder (No-slip U=V=0)
        for i in range(9):
            f_out[i, self.obstacle] = self.f[self.opposite[i], self.obstacle]

        # 4. Streaming
        for i in range(9):
            self.f[i] = np.roll(
                np.roll(f_out[i], self.c[i, 0], axis=0), self.c[i, 1], axis=1
            )

        # 5. Domain Boundaries

        # Top and bottom walls (No-slip U=V=0)
        for i in range(9):
            self.f[i, :, 0] = self.f[self.opposite[i], :, 0]  # Bottom wall (y=0)
            self.f[i, :, -1] = self.f[self.opposite[i], :, -1]  # Top wall (y=-1)
        self.u[:, 0] = 0
        self.v[:, 0] = 0
        self.u[:, -1] = 0
        self.v[:, -1] = 0

        # Obstacle (No-slip U=V=0) applied again after streaming for macroscopic consistency
        self.u[self.obstacle] = 0
        self.v[self.obstacle] = 0

        # OUTLET BOUNDARY CONDITION (Zero-gradient/Neumann: copy from interior)
        self.u[-1, :] = self.u[-2, :]
        self.v[-1, :] = self.v[-2, :]
        self.rho[-1, :] = self.rho[-2, :]
        self.f[:, -1, :] = self.f[:, -2, :]

        # INLET BOUNDARY CONDITION (Velocity Inlet: Constant velocity, extrapolate rho)
        self.u[0, :] = self.u_inlet
        self.v[0, :] = 0
        self.rho[0, :] = self.rho[1, :]
        self.f[:, 0, :] = self.equilibrium(self.rho[0, :], self.u[0, :], self.v[0, :])

    def visualize(self, step):
        """Visualize the velocity field and vorticity."""
        u_magnitude = np.sqrt(self.u**2 + self.v**2)
        plt.figure(figsize=(10, 5))

        # Velocity magnitude
        plt.subplot(1, 2, 1)
        plt.title(f"Velocity Magnitude (Step {step})")
        plt.imshow(u_magnitude.T, origin="lower", cmap="viridis")
        plt.colorbar(label="Velocity")

        # Vorticity
        plt.subplot(1, 2, 2)
        vorticity = np.gradient(self.v, axis=0) - np.gradient(self.u, axis=1)
        plt.title(f"Vorticity (Step {step})")
        plt.imshow(vorticity.T, origin="lower", cmap="RdBu")
        plt.colorbar(label="Vorticity")

        plt.tight_layout()
        plt.show()

    def run_simulation(self, steps):
        """Run the LBM simulation for a given number of steps without animation."""
        for step in range(steps):
            self.step()
            if step % 10 == 0:  # Visualize every 10 steps
                self.visualize(step)

    def run_animation(self, steps=200, interval=50):
        """Run the simulation as an animation."""
        # Using a wider, shorter aspect ratio to prevent excessive whitespace
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # Calculate Reynolds number (Lattice units)
        D_lattice = 2 * (0.05 / self.resolution)  # Diameter = 2 * radius
        nu_lattice = (self.tau - 0.5) / 3.0
        Re = (self.u_inlet * D_lattice) / nu_lattice

        # Add an overall title with the Reynolds number
        fig.suptitle(
            f"LBM Flow Past a Cylinder (Re ≈ {Re:.1f})", fontsize=16, fontweight="bold"
        )

        # Calculate physical extent: [xmin, xmax, ymin, ymax]
        physical_extent = [0, self.nx * self.resolution, 0, self.ny * self.resolution]

        # Initialize plots
        u_magnitude = np.sqrt(self.u**2 + self.v**2)
        vorticity = np.gradient(self.v, axis=0) - np.gradient(self.u, axis=1)

        # Plot 1: Velocity
        vel_plot = ax1.imshow(
            u_magnitude.T, origin="lower", cmap="viridis", extent=physical_extent
        )
        ax1.set_title("Velocity Magnitude")
        ax1.set_xlabel("Length (m)")
        ax1.set_ylabel("Height (m)")
        cbar1 = plt.colorbar(
            vel_plot,
            ax=ax1,
            orientation="horizontal",
            label="Velocity",
            shrink=0.8,
            aspect=40,
            pad=0.18,
        )
        cbar1.ax.tick_params(labelsize=9)

        # Plot 2: Vorticity
        vort_plot = ax2.imshow(
            vorticity.T, origin="lower", cmap="RdBu", extent=physical_extent
        )
        ax2.set_title("Vorticity")
        ax2.set_xlabel("Length (m)")
        ax2.set_ylabel("Height (m)")
        cbar2 = plt.colorbar(
            vort_plot,
            ax=ax2,
            orientation="horizontal",
            label="Vorticity",
            shrink=0.8,
            aspect=40,
            pad=0.18,
        )
        cbar2.ax.tick_params(labelsize=9)

        # Add step counter
        step_text = fig.text(0.5, 0.92, f"Step: 0", ha="center", fontsize=12)

        def update(frame):
            for _ in range(
                5
            ):  # Perform multiple steps per frame for smoother animation
                self.step()

            # Update data
            u_magnitude = np.sqrt(self.u**2 + self.v**2)
            vorticity = np.gradient(self.v, axis=0) - np.gradient(self.u, axis=1)

            vel_plot.set_data(u_magnitude.T)
            vort_plot.set_data(vorticity.T)

            # Dynamically update color scale limits
            vel_plot.set_clim(vmin=0, vmax=np.max(u_magnitude) + 1e-5)
            vmax_vort = np.max(np.abs(vorticity)) + 1e-5
            vort_plot.set_clim(vmin=-vmax_vort, vmax=vmax_vort)

            # Update step counter
            step_text.set_text(f"Step: {frame * 5}")

            return vel_plot, vort_plot, step_text

        print(f"Starting animation with interval={interval} ms...")
        ani = animation.FuncAnimation(
            fig, update, frames=steps, interval=interval, blit=False, repeat=False
        )
        # Adjust layout to make room for the suptitle and horizontal colorbars
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.5)
        plt.show()
        print("Animation completed.")
