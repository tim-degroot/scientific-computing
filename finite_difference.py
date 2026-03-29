import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class finite_difference:
    def __init__(self, Re, u_inlet, resolution):
        # Fixed domain dimensions from the image
        domain_width = 2.2  # meters
        domain_height = 0.41  # meters

        self.resolution = resolution  # STORE RESOLUTION FOR PLOTTING

        self.Re = Re
        self.rho = 1.0
        self.nu = (u_inlet * 2 * 0.05) / self.Re

        self.inlet = u_inlet
        self.dx = resolution
        self.dy = resolution
        self.dt = 0.001

        self.nit = 200
        self.time = 0.0

        # History trackers
        self.time_history = []
        self.max_u_history = []
        self.min_p_history = []
        self.drag_history = []
        self.lift_history = []

        # Calculate grid size based on resolution
        self.nx = int(domain_width / resolution)
        self.ny = int(domain_height / resolution)

        # Initialize macroscopic variables
        self.p = np.zeros((self.nx, self.ny), dtype=float)
        self.u = np.zeros((self.nx, self.ny), dtype=float)
        self.v = np.zeros((self.nx, self.ny), dtype=float)

        # Set inlet velocity profile
        self.u[0,:] = self.inlet

        # Cylinder geometry mapping based on the 2.2m x 0.41m domain setup
        self.obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        cx, cy, r = (
            int(0.20 / resolution),
            int(0.20 / resolution),
            int(0.05 / resolution),
        )
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - cx) ** 2 + (j - cy) ** 2 <= r**2:
                    self.obstacle[i, j] = True

    def update_pressure(self):
        b = np.empty_like(self.p)
        b[1:-1, 1:-1] = (self.rho * 
                         ((1 / self.dt) * 
                          ((self.u[2:,1:-1] - self.u[0:-2, 1:-1]) / (2 * self.dx) +
                           (self.v[1:-1, 2:] - self.v[1:-1, 0:-2]) / (2 * self.dy)) - 
                          np.square((self.u[2:, 1:-1] - self.u[0:-2, 1:-1]) / (2 * self.dx)) -
                           2 * ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dy) *
                                (self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dx)) -
                          np.square((self.v[1:-1, 2:] - self.v[1:-1, 0:-2]) / (2 * self.dy))
                         ))
            
        for _ in range(self.nit):
            n_p = np.copy(self.p)
            self.p[1:-1, 1:-1] = (
                ((n_p[2:, 1:-1] + n_p[:-2, 1:-1]) * self.dy**2 + 
                 (n_p[1:-1, 2:] + n_p[1:-1, :-2]) * self.dx**2) /
                 (2 * (self.dx**2 + self.dy**2)) -
                 (self.dx**2 * self.dy**2) / (2 * (self.dx**2 + self.dy**2)) * 
                 b[1:-1, 1:-1])

            # Boundary conditions
            self.p[0, :] = self.p[1, :]     # inlet
            self.p[-1, :] = 0               # outlet
            self.p[:, 0] = self.p[:, 1]
            self.p[:, -1] = self.p[:, -2]
            
        return self.p
    
    def compute_forces(self):
        drag = 0.0
        lift = 0.0

        min_bound = int(0.10 / self.resolution)
        max_bound = int(0.30 / self.resolution)

        for i in range(min_bound, max_bound):
            for j in range(min_bound, max_bound):
                if self.obstacle[i, j]:
                    continue

                # Check neighbors → surface detection
                neighbors = [
                    (i+1, j, 1, 0),
                    (i-1, j, -1, 0),
                    (i, j+1, 0, 1),
                    (i, j-1, 0, -1),
                ]

                for ni, nj, nx, ny in neighbors:
                    if self.obstacle[ni, nj]:
                        # Normal vector points INTO fluid
                        p = self.p[i, j]

                        # Pressure contribution
                        drag += -p * nx * self.dy
                        lift += -p * ny * self.dx

                        # Viscous contribution (simplified shear)
                        du_dn = (self.u[i, j] - self.u[ni, nj]) / self.dx
                        dv_dn = (self.v[i, j] - self.v[ni, nj]) / self.dy

                        drag += self.nu * du_dn * self.dy
                        lift += self.nu * dv_dn * self.dx

        return drag, lift


    def step(self):
        n_u = self.u.copy()
        n_v = self.v.copy()

        p = self.update_pressure()

        # Derivatives (upwind method)
        du_dx = np.where(n_u > 0,
                        (n_u - np.roll(n_u, 1, axis=0)) / self.dx,
                        (np.roll(n_u, -1, axis=0) - n_u) / self.dx)

        du_dy = np.where(n_v > 0,
                        (n_u - np.roll(n_u, 1, axis=1)) / self.dy,
                        (np.roll(n_u, -1, axis=1) - n_u) / self.dy)

        dv_dx = np.where(n_u > 0,
                        (n_v - np.roll(n_v, 1, axis=0)) / self.dx,
                        (np.roll(n_v, -1, axis=0) - n_v) / self.dx)

        dv_dy = np.where(n_v > 0,
                        (n_v - np.roll(n_v, 1, axis=1)) / self.dy,
                        (np.roll(n_v, -1, axis=1) - n_v) / self.dy)
                        

        # Diffusion
        d2u_dx2 = (np.roll(n_u, -1, axis=0) - 2*n_u + np.roll(n_u, 1, axis=0)) / self.dx**2
        d2u_dy2 = (np.roll(n_u, -1, axis=1) - 2*n_u + np.roll(n_u, 1, axis=1)) / self.dy**2

        d2v_dx2 = (np.roll(n_v, -1, axis=0) - 2*n_v + np.roll(n_v, 1, axis=0)) / self.dx**2
        d2v_dy2 = (np.roll(n_v, -1, axis=1) - 2*n_v + np.roll(n_v, 1, axis=1)) / self.dy**2

        # Pressure gradients
        dp_dx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2*self.dx)
        dp_dy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2*self.dy)

        # Velocity update
        self.u = n_u - self.dt * (n_u * du_dx + n_v * du_dy) \
                - self.dt / self.rho * dp_dx \
                + self.nu * self.dt * (d2u_dx2 + d2u_dy2)

        self.v = n_v - self.dt * (n_u * dv_dx + n_v * dv_dy) \
                - self.dt / self.rho * dp_dy \
                + self.nu * self.dt * (d2v_dx2 + d2v_dy2)

        # Boundary conditions
        # inlet
        self.u[0, :] = self.inlet
        self.v[0, :] = 0

        # outlet
        self.u[-1, :] = self.u[-2, :]
        self.v[-1, :] = self.v[-2, :]

        # walls
        self.u[:, 0] = 0
        self.u[:, -1] = 0
        self.v[:, 0] = 0
        self.v[:, -1] = 0

        # obstacle
        self.u[self.obstacle] = 0
        self.v[self.obstacle] = 0

        self.time += self.dt

        # Track histories
        self.time_history.append(self.time)

        u_mag = np.sqrt(self.u**2 + self.v**2)
        p_masked = np.where(self.obstacle, np.nan, self.p)
        drag, lift = self.compute_forces()

        self.max_u_history.append(np.max(u_mag))
        self.min_p_history.append(np.nanmin(p_masked))
        self.drag_history.append(drag)
        self.lift_history.append(lift)

        return self.u, self.v, self.p
        
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
        for step in range(steps):
            self.step()
            if step % 10 == 0:  # Visualize every 10 steps
                self.visualize(step)
    
    def run(self, steps, animate=False, interval=5, steps_per_frame=1, store_every=None):
        # Reset histories for each run
        self.time_history = [0]
        self.max_u_history = [1]
        self.min_p_history = [0]
        self.drag_history = [0]
        self.lift_history = [0]

        if not animate:
            print(f"Running simulation for {steps} steps...")
            for step in range(steps):
                self.step()
                if isinstance(store_every, int) and (step + 1) % store_every == 0:
                    self.visualize(step + 1)
            print("Simulation completed.")
            return

        """ Animation mode """
        # Using a wider, shorter aspect ratio to prevent excessive whitespace
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # Add an overall title with the Reynolds number
        fig.suptitle(
            f"Flow Past a Cylinder (Re ≈ {self.Re:.1f}), u={self.inlet:.2f})", fontsize=16, fontweight="bold"
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
        plt.colorbar(vel_plot, ax=ax1, orientation="horizontal", shrink=0.8, pad=0.18,)

        # Plot 2: Vorticity
        vort_plot = ax2.imshow(
            vorticity.T, origin="lower", cmap="RdBu", extent=physical_extent
        )
        ax2.set_title("Vorticity")
        plt.colorbar(vort_plot, ax=ax2, orientation="horizontal", shrink=0.8, pad=0.18)

        # Add time counter
        step_text = fig.text(0.5, 0.92, f"Time: 0.000 s", ha="center", fontsize=12)

        def update(frame):
            for _ in range(steps_per_frame):
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
            step_text.set_text(f"Time: {self.time:.3f} s")

            return vel_plot, vort_plot, step_text

        print(f"Starting animation with interval={interval} ms...")
        ani = animation.FuncAnimation(
            fig, update, frames=int(steps/steps_per_frame), interval=interval, blit=False, repeat=False
        )
        # Adjust layout to make room for the suptitle and horizontal colorbars
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.5)
        plt.show()
        print("Animation completed.")
    
    def plot_histories(self, show=True, save_path=None):
        """
        Plot time histories of max velocity, min pressure, lift, and drag.
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(r"FDM Simulation: Physical Quantities over Time")

        # Left: Max Absolute Velocity
        axs[0,0].plot(self.time_history, self.max_u_history, color="tab:red")
        axs[0,0].set_title(r"Max Absolute Velocity ($|\mathbf{u}|_{\max}$)")
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_ylabel(r"Velocity ($|\mathbf{u}|$) (lu/ts)")
        axs[0,0].grid(True, alpha=0.3)

        # Right: Min Pressure
        axs[0,1].plot(self.time_history, self.min_p_history, color="tab:blue")
        axs[0,1].set_title(r"Minimum Pressure ($p_{\min}$)")
        axs[0,1].set_xlabel("Time")
        axs[0,1].set_ylabel("Pressure (lu)")
        axs[0,1].grid(True, alpha=0.3)

        # Bottom Left: Lift (Accuracy/Shedding)
        axs[1, 0].plot(self.lift_history, color="tab:orange")
        axs[1, 0].set_title(r"Lift Force ($F_L$)")
        axs[1, 0].set_xlabel("Time")
        axs[1, 0].set_ylabel(r"$F_L$ (lu)")
        axs[1, 0].grid(True, alpha=0.3)

        # Bottom Right: Drag (Stability)
        axs[1, 1].plot(self.drag_history, color="tab:green")
        axs[1, 1].set_title(r"Drag Force ($F_D$)")
        axs[1, 1].set_xlabel("Time")
        axs[1, 1].set_ylabel(r"$F_D$ (lu)")
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()

        
experiments = [
        {"resolution": 0.005, "Re": 100, "u_inlet": 1, "steps": 10000},
        # {"resolution": 0.005, "Re": 150, "u_inlet": 1, "steps": 10000},
        # {"resolution": 0.005, "Re": 200, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 250, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 500, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 750, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 1000, "u_inlet": 1, "steps": 10000},
        # {"resolution": 0.005, "Re": 1500, "u_inlet": 1, "steps": 10000},
        # {"resolution": 0.005, "Re": 5000, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 10000, "u_inlet": 1, "steps": 10000},
    ]


for i, params in enumerate(experiments):
    resolution, Re, u_inlet, steps = params.values()
    print(f"\nRunning experiment {i+1} with Re={Re}, u_inlet={u_inlet}")

    # Instantiate model
    fdm = finite_difference(Re, u_inlet, resolution)

    # Unified runner call
    fdm.run(steps=steps, animate=True, interval=5, steps_per_frame=10)
    print("   Simulation completed.")

    fdm.plot_histories(show=True)
