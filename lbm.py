import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class D2Q9LBM:
    """
    Lattice Boltzmann Method (D2Q9) for flow past a cylinder in a 2D channel.
    Domain: 2.2m x 0.41m, cylinder at (0.15, 0.21) with radius 0.05m.
    """

    def __init__(self, resolution, Re, u_inlet):
        # Domain size (meters)
        width, height = 2.2, 0.41
        self.resolution = resolution
        self.nx = int(width / resolution)
        self.ny = int(height / resolution)
        self.Re = Re
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

        # Cylinder geometry
        cx_cyl = 0.2 / self.resolution    # cylinder center x
        cy_cyl = 0.2 / self.resolution    # cylinder center y
        r_cyl  = 0.05 / self.resolution    # cylinder radius

        # Boolean array: True where the obstacle is located
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')    # X,Y have shape (Nx, Ny)
        self.obstacle = (X - cx_cyl)**2 + (Y - cy_cyl)**2 <= r_cyl**2
        
        # Derived quantities:
        #   Re = U * D / nu   →  nu = U * D / Re
        #   In LBM:  nu = cs² * (tau - 0.5)  where cs² = 1/3
        #   Therefore:  tau = 3 * nu + 0.5
        D   = 2 * r_cyl                             # cylinder diameter
        self.nu  = self.u_inlet * D / self.Re                      # kinematic viscosity
        self.tau = 3.0 * self.nu + 0.5                        # BGK relaxation time

        # History trackers
        self.drag_history = []
        self.lift_history = []
        self.max_u_history = []
        self.min_p_history = []
        
        # Fields
        self.rho = np.ones((self.nx, self.ny))
        self.ux = np.full((self.nx, self.ny), self.u_inlet)
        self.uy = np.zeros((self.nx, self.ny))
        # self.uy[0, :] = u_inlet
        
        # Small transverse perturbation to break symmetry and trigger vortex shedding
        # self.uy += 0.001 * self.u_inlet * np.sin(2.0 * np.pi * Y / self.ny)
        
        # Set velocity to zero inside the obstacle
        self.ux[self.obstacle] = 0.0
        self.uy[self.obstacle] = 0.0
        
        self.f = self.equilibrium()

    def equilibrium(self):
        """
        Compute the equilibrium distribution f^eq for the D2Q9 lattice.

        The equilibrium is derived from a second-order Taylor expansion of the
        Maxwell-Boltzmann distribution:

            f_i^eq = w_i * rho * (1 + c_i·u/cs² + (c_i·u)²/(2·cs⁴) - u·u/(2·cs²))

        where cs² = 1/3  (lattice speed of sound squared).

        Parameters
        ----------
        rho : ndarray (Nx, Ny)   — macroscopic density
        ux  : ndarray (Nx, Ny)   — x-component of velocity
        uy  : ndarray (Nx, Ny)   — y-component of velocity

        Returns
        -------
        feq : ndarray (Nx, Ny, 9) — equilibrium distributions
        """
        feq = np.zeros((self.nx, self.ny, 9))
        usqr = self.ux**2 + self.uy**2                 # |u|²

        for i in range(9):
            cu = self.c[i, 0] * self.ux + self.c[i, 1] * self.uy  # c_i · u
            feq[:, :, i] = self.w[i] * self.rho * (1.0
                                        + 3.0 * cu            # c_i·u / cs²
                                        + 4.5 * cu**2         # (c_i·u)² / (2·cs⁴)
                                        - 1.5 * usqr)         # -|u|² / (2·cs²)
        return feq

    def step(self):
        """
        Advance the simulation by one time step (collision, streaming, boundaries).
        """
        # -------------------------------------------------------------
        # 7a.  Macroscopic quantities: density and velocity
        #      rho = Σ f_i,   rho·u = Σ c_i · f_i
        # -------------------------------------------------------------
        self.rho = np.sum(self.f, axis=2)
        self.ux  = np.sum(self.f * self.c[:, 0], axis=2) / self.rho
        self.uy  = np.sum(self.f * self.c[:, 1], axis=2) / self.rho


        # Track histories
        u_mag = np.sqrt(self.ux**2 + self.uy**2)
        self.max_u_history.append(np.max(u_mag))
        
        # Mask cylinder area as NaN to exclude it from minimum pressure tracking
        rho_masked = np.where(self.obstacle, np.nan, self.rho)
        self.min_p_history.append(np.nanmin(rho_masked) / 3.0)

        # Collision (BGK)
        feq = self.equilibrium()
        f_out = self.f - (self.f - feq) / self.tau


        # Bounce-back and force tracking
        # Cylinder
        drag, lift = 0.0, 0.0
        for i in range(9):
            f_out[self.obstacle, i] = self.f[self.obstacle, self.opposite[i]]
            
            # Compute drag and lift coefficient
            dp_x = 2 * self.f[self.obstacle, self.opposite[i]] * self.c[i, 0]
            dp_y = 2 * self.f[self.obstacle, self.opposite[i]] * self.c[i, 1]
            drag += np.sum(dp_x)
            lift += np.sum(dp_y)
        # Track drag and lift coefficient
        self.drag_history.append(drag)
        self.lift_history.append(lift)
        
        # Streaming
        for i in range(9):
            self.f[:, :, i] = np.roll(f_out[:, :, i], shift=self.c[i, 0], axis=0)
            self.f[:, :, i] = np.roll(self.f[:, :, i],shift=self.c[i, 1], axis=1)
            
        # Apply this to the WHOLE length of the top/bottom rows
        self.f[:, 0, [2,5,6]] = f_out[:, 0, [4,7,8]]
        self.f[:, -1, [4,7,8]] = f_out[:, -1, [2,5,6]]
        
        # -------------------------------------------------------------
        # 7f.  Inlet boundary condition (Zou-He, fixed velocity)
        #      After streaming, populations 1, 5, 8 at x=0 are unknown
        #      (they would come from outside the domain).  Zou-He
        #      determines them from known populations and prescribed
        #      inlet velocity (ux=U_inlet, uy=0).
        # -------------------------------------------------------------
        f = self.f # shorthand
        rho_in = (f[0,:,0] + f[0,:,2] + f[0,:,4] + 2*(f[0,:,3] + f[0,:,6] + f[0,:,7])) / (1 - self.u_inlet)
        
        self.f[0,:,1] = f[0,:,3] + (2/3)*rho_in*self.u_inlet
        self.f[0,:,5] = f[0,:,7] - 0.5*(f[0,:,2] - f[0,:,4]) + (1/6)*rho_in*self.u_inlet
        self.f[0,:,8] = f[0,:,6] + 0.5*(f[0,:,2] - f[0,:,4]) + (1/6)*rho_in*self.u_inlet
        
        # Outlet (right)
        self.f[-1, :, :] = self.f[-2, :, :]
        

    def visualize(self, step):
        """
        Show static velocity and vorticity plots for a given step.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(
            f"LBM Flow Past Cylinder (tau={self.tau}, u={self.u_inlet}, Re={self.Re:.1f})"
        )
        extent = [0, self.nx * self.resolution, 0, self.ny * self.resolution]
        u_mag = np.sqrt(self.ux**2 + self.uy**2)
        vort = np.gradient(self.uy, axis=0) - np.gradient(self.ux, axis=1)
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
        u_mag = np.sqrt(self.ux**2 + self.uy**2)
        drag, lift = 0.0, 0.0
        for i in range(9):
            dp_x = 2 * self.f[self.obstacle, self.opposite[i]] * self.c[i, 0]
            dp_y = 2 * self.f[self.obstacle, self.opposite[i]] * self.c[i, 1]
            drag += np.sum(dp_x)
            lift += np.sum(dp_y)
        self.max_u_history.append(np.max(u_mag))
        self.drag_history.append(drag)
        self.lift_history.append(lift)
        
        # Mask cylinder area as NaN to exclude it from minimum pressure tracking
        rho_masked = np.where(self.obstacle, np.nan, self.rho)
        self.min_p_history.append(np.nanmin(rho_masked) / 3.0)
        
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
        fig.suptitle(f"LBM Flow (tau={self.tau}, u={self.u_inlet}, Re={self.Re:.1f})")
        extent = [0, self.nx * self.resolution, 0, self.ny * self.resolution]
        u_mag = np.sqrt(self.ux**2 + self.uy**2)
        vort = np.gradient(self.uy, axis=0) - np.gradient(self.ux, axis=1)
        im1 = ax1.imshow(u_mag.T, origin="lower", cmap="viridis", extent=extent)
        ax1.set_title("Velocity Magnitude")
        plt.colorbar(im1, ax=ax1, orientation="horizontal", shrink=0.8, pad=0.18)
        im2 = ax2.imshow(vort.T, origin="lower", cmap="RdBu", extent=extent)
        ax2.set_title("Vorticity")
        plt.colorbar(im2, ax=ax2, orientation="horizontal", shrink=0.8, pad=0.18)
        step_text = fig.text(0.5, 0.92, f"Step: 0", ha="center", fontsize=12)

        def update(frame):
            self.step()
            u_mag = np.sqrt(self.ux**2 + self.uy**2)
            vort = np.gradient(self.uy, axis=0) - np.gradient(self.ux, axis=1)
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
        axs[1, 0].set_title(r"Lift Force ($F_L$)")
        axs[1, 0].set_xlabel("Time Step")
        axs[1, 0].set_ylabel(r"$F_L$ (lu)")
        axs[1, 0].grid(True, alpha=0.3)

        # Bottom Right: Drag (Stability)
        axs[1, 1].plot(self.drag_history, color="tab:green")
        axs[1, 1].set_title(r"Drag Force ($F_D$)")
        axs[1, 1].set_xlabel("Time Step")
        axs[1, 1].set_ylabel(r"$F_D$ (lu)")
        axs[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()