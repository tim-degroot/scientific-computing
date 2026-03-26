import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class D2Q9LBM:
    def __init__(self, resolution, tau, u_inlet):
        # Fixed domain dimensions from the image
        domain_width = 2.2  # meters
        domain_height = 0.41  # meters

        self.resolution = resolution

        # Calculate grid size based on resolution
        self.nx = int(domain_width / resolution)
        self.ny = int(domain_height / resolution)
        self.tau = max(0.51, tau)
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

        # --- HISTORY TRACKERS ---
        self.drag_history = []
        self.lift_history = []
        self.max_u_history = []
        self.min_p_history = []

    def equilibrium(self, rho, u, v):
        """Calculate the equilibrium distribution."""
        feq = np.zeros((9,) + rho.shape)
        u2 = u**2 + v**2
        for i in range(9):
            cu = self.c[i, 0] * u + self.c[i, 1] * v
            cu = np.clip(cu, -1e3, 1e3)
            feq[i] = self.w[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return np.clip(feq, 0, 1e3)

    def step(self):
        """Perform one LBM time step (Collision + Streaming)."""
        # 1. Macroscopic variables
        self.rho = np.sum(self.f, axis=0)
        self.rho = np.clip(self.rho, 1e-3, 1e3)
        self.u = (
            np.sum(self.f * self.c[:, 0, np.newaxis, np.newaxis], axis=0) / self.rho
        )
        self.v = (
            np.sum(self.f * self.c[:, 1, np.newaxis, np.newaxis], axis=0) / self.rho
        )

        # Track Max Velocity and Min Pressure (P = rho / 3 in LBM)
        u_mag = np.sqrt(self.u**2 + self.v**2)
        self.max_u_history.append(np.max(u_mag))
        self.min_p_history.append(np.min(self.rho) / 3.0)

        # 2. Collision (BGK approximation)
        feq = self.equilibrium(self.rho, self.u, self.v)
        f_out = self.f - (self.f - feq) / self.tau

        # 3. Bounce-back boundary for the cylinder & Momentum Exchange for Forces
        step_drag = 0.0
        step_lift = 0.0

        for i in range(9):
            # Calculate momentum exchange (drag & lift) from fluid hitting the obstacle
            dp_x = 2 * self.f[self.opposite[i], self.obstacle] * self.c[i, 0]
            dp_y = 2 * self.f[self.opposite[i], self.obstacle] * self.c[i, 1]
            step_drag += np.sum(dp_x)
            step_lift += np.sum(dp_y)

            # Apply standard bounce-back
            f_out[i, self.obstacle] = self.f[self.opposite[i], self.obstacle]

        self.drag_history.append(step_drag)
        self.lift_history.append(step_lift)

        # 4. Streaming
        for i in range(9):
            self.f[i] = np.roll(
                np.roll(f_out[i], self.c[i, 0], axis=0), self.c[i, 1], axis=1
            )

        # 5. Domain Boundaries
        for i in range(9):
            self.f[i, :, 0] = self.f[self.opposite[i], :, 0]  # Bottom
            self.f[i, :, -1] = self.f[self.opposite[i], :, -1]  # Top

        self.u[:, 0] = 0
        self.v[:, 0] = 0
        self.u[:, -1] = 0
        self.v[:, -1] = 0

        self.u[self.obstacle] = 0
        self.v[self.obstacle] = 0

        # OUTLET BOUNDARY CONDITION
        self.u[-1, :] = self.u[-2, :]
        self.v[-1, :] = self.v[-2, :]
        self.rho[-1, :] = self.rho[-2, :]
        self.f[:, -1, :] = self.f[:, -2, :]

        # INLET BOUNDARY CONDITION
        self.u[0, :] = self.u_inlet
        self.v[0, :] = 0
        self.rho[0, :] = self.rho[1, :]
        self.f[:, 0, :] = self.equilibrium(self.rho[0, :], self.u[0, :], self.v[0, :])

    def visualize(self, step):
        """Static visualization for headless runs."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        D_lattice = 2 * (0.05 / self.resolution)
        nu_lattice = (self.tau - 0.5) / 3.0
        Re = (self.u_inlet * D_lattice) / nu_lattice

        fig.suptitle(
            rf"LBM Flow Past a Cylinder ($\tau$={self.tau}, $u$={self.u_inlet}, Re = {Re:.1f})"
        )
        physical_extent = [0, self.nx * self.resolution, 0, self.ny * self.resolution]

        u_magnitude = np.sqrt(self.u**2 + self.v**2)
        vorticity = np.gradient(self.v, axis=0) - np.gradient(self.u, axis=1)

        vel_plot = ax1.imshow(
            u_magnitude.T, origin="lower", cmap="viridis", extent=physical_extent
        )
        ax1.set_title("Velocity Magnitude")
        plt.colorbar(vel_plot, ax=ax1, orientation="horizontal", shrink=0.8, pad=0.18)

        vort_plot = ax2.imshow(
            vorticity.T, origin="lower", cmap="RdBu", extent=physical_extent
        )
        ax2.set_title("Vorticity")
        plt.colorbar(vort_plot, ax=ax2, orientation="horizontal", shrink=0.8, pad=0.18)

        fig.text(0.5, 0.92, f"Step: {step}", ha="center", fontsize=12)
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.5)
        plt.show()

    def run(self, steps, animate=False, interval=50, store_every=None):
        """Unified runner for both standard execution and animations."""
        if not animate:
            print(f"Running simulation quietly for {steps} steps...")
            for step in range(steps):
                self.step()
                if isinstance(store_every, int) and (step + 1) % store_every == 0:
                    self.visualize(step + 1)
            print("Simulation completed.")
            return

        # --- ANIMATION SETUP ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        D_lattice = 2 * (0.05 / self.resolution)
        nu_lattice = (self.tau - 0.5) / 3.0
        Re = (self.u_inlet * D_lattice) / nu_lattice

        fig.suptitle(
            rf"LBM Flow ($\tau$={self.tau}, $u=${self.u_inlet}, Re = {Re:.1f})"
        )
        physical_extent = [0, self.nx * self.resolution, 0, self.ny * self.resolution]

        u_magnitude = np.sqrt(self.u**2 + self.v**2)
        vorticity = np.gradient(self.v, axis=0) - np.gradient(self.u, axis=1)

        vel_plot = ax1.imshow(
            u_magnitude.T, origin="lower", cmap="viridis", extent=physical_extent
        )
        ax1.set_title("Velocity Magnitude")
        plt.colorbar(vel_plot, ax=ax1, orientation="horizontal", shrink=0.8, pad=0.18)

        vort_plot = ax2.imshow(
            vorticity.T, origin="lower", cmap="RdBu", extent=physical_extent
        )
        ax2.set_title("Vorticity")
        plt.colorbar(vort_plot, ax=ax2, orientation="horizontal", shrink=0.8, pad=0.18)

        step_text = fig.text(0.5, 0.92, f"Step: 0", ha="center", fontsize=12)

        def update(frame):
            self.step()
            u_mag = np.sqrt(self.u**2 + self.v**2)
            vort = np.gradient(self.v, axis=0) - np.gradient(self.u, axis=1)

            vel_plot.set_data(u_mag.T)
            vort_plot.set_data(vort.T)

            vel_plot.set_clim(vmin=0, vmax=np.max(u_mag) + 1e-5)
            vmax_vort = np.max(np.abs(vort)) + 1e-5
            vort_plot.set_clim(vmin=-vmax_vort, vmax=vmax_vort)
            step_text.set_text(f"Step: {frame + 1}")

            return vel_plot, vort_plot, step_text

        print(f"Starting animation with interval={interval} ms...")
        ani = animation.FuncAnimation(
            fig, update, frames=steps, interval=interval, blit=False, repeat=False
        )
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.5)
        plt.show()
        print("Animation completed.")

    def plot_histories(self, show=True, save_path=None):
        """Plot the tracked physical quantities with formal titles and strict units."""
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(r"LBM Simulation: Physical Quantities over Time")

        # Top Left: Max Absolute Velocity
        axs[0, 0].plot(self.max_u_history, color="tab:red")
        axs[0, 0].set_title(r"Max Absolute Velocity ($|\mathbf{u}|_{\max}$)")
        axs[0, 0].set_xlabel("Time Step")
        axs[0, 0].set_ylabel(r"Velocity ($|\mathbf{u}|$) (lu/ts)")
        axs[0, 0].grid(True, alpha=0.3)

        # Top Right: Min Pressure
        axs[0, 1].plot(self.min_p_history, color="tab:blue")
        axs[0, 1].set_title(
            r"Minimum Pressure ($p_{\min}$)"
        )  # Switched to small p for local pressure
        axs[0, 1].set_xlabel("Time Step")
        axs[0, 1].set_ylabel("Pressure (lu)")
        axs[0, 1].grid(True, alpha=0.3)

        # Bottom Left: Lift (Accuracy/Shedding)
        axs[1, 0].plot(self.lift_history, color="tab:orange")
        axs[1, 0].set_title(r"Lift Coefficient ($C_L$)")
        axs[1, 0].set_xlabel("Time Step")
        axs[1, 0].set_ylabel(r"$C_L$")
        axs[1, 0].grid(True, alpha=0.3)

        # Bottom Right: Drag (Stability)
        axs[1, 1].plot(self.drag_history, color="tab:green")
        axs[1, 1].set_title(r"Drag Coefficient ($C_D$)")
        axs[1, 1].set_xlabel("Time Step")
        axs[1, 1].set_ylabel(r"$C_D$")
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        if show:
            plt.show()
