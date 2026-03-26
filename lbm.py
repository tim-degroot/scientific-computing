import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class D2Q9LBM:
    """
    Lattice Boltzmann Method (D2Q9) for flow past a cylinder in a 2D channel.
    Domain: 2.2m x 0.41m, cylinder at (0.15, 0.21) with radius 0.05m.
    """

    def __init__(self, resolution, tau, u_inlet):
        # Domain size (meters)
        width, height = 2.2, 0.41
        self.resolution = resolution
        self.nx = int(width / resolution)
        self.ny = int(height / resolution)
        self.tau = max(0.51, tau)
        self.u_inlet = u_inlet

        # D2Q9 lattice weights and directions
        self.w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
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
        self.opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # Fields
        self.rho = np.ones((self.nx, self.ny))
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.u[0, :] = u_inlet
        self.f = self.equilibrium(self.rho, self.u, self.v)

        # Cylinder obstacle mask
        self.obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        cx, cy, r = (
            int(0.15 / resolution),
            int(0.20 / resolution),
            int(0.05 / resolution),
        )
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - cx) ** 2 + (j - cy) ** 2 <= r**2:
                    self.obstacle[i, j] = True

        # History trackers
        self.drag_history = []
        self.lift_history = []
        self.max_u_history = []
        self.min_p_history = []

    def equilibrium(self, rho, u, v):
        """
        Compute equilibrium distribution for given density and velocity fields.
        """
        feq = np.zeros((9,) + rho.shape)
        u2 = u**2 + v**2
        for i in range(9):
            cu = self.c[i, 0] * u + self.c[i, 1] * v
            feq[i] = self.w[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return feq

    def step(self):
        """
        Advance the simulation by one time step (collision, streaming, boundaries).
        """
        # Macroscopic variables
        self.rho = np.sum(self.f, axis=0)
        self.u = np.sum(self.f * self.c[:, 0, None, None], axis=0) / self.rho
        self.v = np.sum(self.f * self.c[:, 1, None, None], axis=0) / self.rho

        # Track histories
        u_mag = np.sqrt(self.u**2 + self.v**2)
        self.max_u_history.append(np.max(u_mag))
        self.min_p_history.append(np.min(self.rho) / 3.0)

        # Collision (BGK)
        feq = self.equilibrium(self.rho, self.u, self.v)
        f_post = self.f - (self.f - feq) / self.tau

        # Bounce-back and force tracking
        drag, lift = 0.0, 0.0
        for i in range(9):
            dp_x = 2 * self.f[self.opposite[i], self.obstacle] * self.c[i, 0]
            dp_y = 2 * self.f[self.opposite[i], self.obstacle] * self.c[i, 1]
            drag += np.sum(dp_x)
            lift += np.sum(dp_y)
            f_post[i, self.obstacle] = self.f[self.opposite[i], self.obstacle]
        self.drag_history.append(drag)
        self.lift_history.append(lift)

        # Streaming
        for i in range(9):
            self.f[i] = np.roll(
                np.roll(f_post[i], self.c[i, 0], axis=0), self.c[i, 1], axis=1
            )

        # Boundary conditions
        for i in range(9):
            self.f[i, :, 0] = self.f[self.opposite[i], :, 0]  # Bottom
            self.f[i, :, -1] = self.f[self.opposite[i], :, -1]  # Top

        self.u[:, 0] = self.v[:, 0] = 0
        self.u[:, -1] = self.v[:, -1] = 0
        self.u[self.obstacle] = self.v[self.obstacle] = 0


        # Outlet (right)
        self.f[:, -1, :] = self.f[:, -2, :]
        self.u[-1, :] = self.u[-2, :]
        self.v[-1, :] = self.v[-2, :]
        self.rho[-1, :] = self.rho[-2, :]

        # Inlet (left)
        self.u[0, :] = self.u_inlet
        self.v[0, :] = 0
        self.rho[0, :] = self.rho[1, :]
        self.f[:, 0, :] = self.equilibrium(self.rho[0, :], self.u[0, :], self.v[0, :])

    def visualize(self, step):
        """
        Show static velocity and vorticity plots for a given step.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        D = 2 * (0.05 / self.resolution)
        nu = (self.tau - 0.5) / 3.0
        Re = (self.u_inlet * D) / nu
        fig.suptitle(
            f"LBM Flow Past Cylinder (tau={self.tau}, u={self.u_inlet}, Re={Re:.1f})"
        )
        extent = [0, self.nx * self.resolution, 0, self.ny * self.resolution]
        u_mag = np.sqrt(self.u**2 + self.v**2)
        vort = np.gradient(self.v, axis=0) - np.gradient(self.u, axis=1)
        im1 = ax1.imshow(u_mag.T, origin="lower", cmap="viridis", extent=extent)
        ax1.set_title("Velocity Magnitude")
        plt.colorbar(im1, ax=ax1, orientation="horizontal", shrink=0.8, pad=0.18)
        im2 = ax2.imshow(vort.T, origin="lower", cmap="RdBu", extent=extent)
        ax2.set_title("Vorticity")
        plt.colorbar(im2, ax=ax2, orientation="horizontal", shrink=0.8, pad=0.18)
        fig.text(0.5, 0.92, f"Step: {step}", ha="center", fontsize=12)
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.5)
        plt.show()

    def run(self, steps, animate=False, interval=50, store_every=None):
        """
        Run the simulation for a given number of steps, optionally with animation.
        Resets histories at the start of each run.
        """

        # Reset histories for each run
        self.drag_history = []
        self.lift_history = []
        self.max_u_history = []
        self.min_p_history = []

        # Append initial state (t=0)
        u_mag = np.sqrt(self.u**2 + self.v**2)
        drag, lift = 0.0, 0.0
        for i in range(9):
            dp_x = 2 * self.f[self.opposite[i], self.obstacle] * self.c[i, 0]
            dp_y = 2 * self.f[self.opposite[i], self.obstacle] * self.c[i, 1]
            drag += np.sum(dp_x)
            lift += np.sum(dp_y)
        self.max_u_history.append(np.max(u_mag))
        self.min_p_history.append(np.min(self.rho) / 3.0)
        self.drag_history.append(drag)
        self.lift_history.append(lift)

        if not animate:
            print(f"Running simulation for {steps} steps...")
            for step in range(steps):
                self.step()
                if isinstance(store_every, int) and (step + 1) % store_every == 0:
                    self.visualize(step + 1)
            print("Simulation completed.")
            return

        # Animation mode
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        D = 2 * (0.05 / self.resolution)
        nu = (self.tau - 0.5) / 3.0
        Re = (self.u_inlet * D) / nu
        fig.suptitle(f"LBM Flow (tau={self.tau}, u={self.u_inlet}, Re={Re:.1f})")
        extent = [0, self.nx * self.resolution, 0, self.ny * self.resolution]
        u_mag = np.sqrt(self.u**2 + self.v**2)
        vort = np.gradient(self.v, axis=0) - np.gradient(self.u, axis=1)
        im1 = ax1.imshow(u_mag.T, origin="lower", cmap="viridis", extent=extent)
        ax1.set_title("Velocity Magnitude")
        plt.colorbar(im1, ax=ax1, orientation="horizontal", shrink=0.8, pad=0.18)
        im2 = ax2.imshow(vort.T, origin="lower", cmap="RdBu", extent=extent)
        ax2.set_title("Vorticity")
        plt.colorbar(im2, ax=ax2, orientation="horizontal", shrink=0.8, pad=0.18)
        step_text = fig.text(0.5, 0.92, f"Step: 0", ha="center", fontsize=12)

        def update(frame):
            self.step()
            u_mag = np.sqrt(self.u**2 + self.v**2)
            vort = np.gradient(self.v, axis=0) - np.gradient(self.u, axis=1)
            im1.set_data(u_mag.T)
            im2.set_data(vort.T)
            im1.set_clim(vmin=0, vmax=np.max(u_mag) + 1e-5)
            vmax_vort = np.max(np.abs(vort)) + 1e-5
            im2.set_clim(vmin=-vmax_vort, vmax=vmax_vort)
            step_text.set_text(f"Step: {frame + 1}")
            return im1, im2, step_text

        print(f"Starting animation with interval={interval} ms...")
        ani = animation.FuncAnimation(
            fig, update, frames=steps, interval=interval, blit=False, repeat=False
        )
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.5)
        plt.show()
        print("Animation completed.")

    def plot_histories(self, show=True, save_path=None):
        """
        Plot time histories of max velocity, min pressure, lift, and drag.
        """
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
        axs[0, 1].set_title(r"Minimum Pressure ($p_{\min}$)")
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
