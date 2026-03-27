import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class finite_difference:
    def __init__(self, rho, nu, inlet, resolution):
        # Fixed domain dimensions from the image
        domain_width = 2.2  # meters
        domain_height = 0.41  # meters

        self.resolution = resolution  # STORE RESOLUTION FOR PLOTTING

        # Fixed values for density and viscosity
        self.rho = rho
        self.nu = nu

        self.inlet = inlet
        self.dx = resolution
        self.dy = resolution
        # self.dt = 0.001
        self.dt = 0.2 * min(self.dx / self.inlet, self.dx**2 / self.nu)

        self.nit = 100
        self.time = 0.0

        # Calculate grid size based on resolution
        self.nx = int(domain_width / resolution)
        self.ny = int(domain_height / resolution)

        # Initialize macroscopic variables
        self.p = np.ones((self.nx, self.ny))
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))

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
                         (1 / self.dt * 
                          ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx) +
                           (self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy)) - 
                          ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx))**2 -
                           2 * ((self.u[2:, 1:-1] - self.u[0:-2, 1:-1]) / (2 * self.dy) *
                                (self.v[1:-1, 2:] - self.v[1:-1, 0:-2]) / (2 * self.dx)) -
                          ((self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy))**2
                         ))
            
        for _ in range(self.nit):
            n_p = np.copy(self.p)
            self.p[1:-1, 1:-1] = (
                ((n_p[1:-1, 2:] + n_p[1:-1, :-2]) * self.dy**2 + 
                 (n_p[2:, 1:-1] + n_p[:-2, 1:-1]) * self.dx**2) /
                 (2 * (self.dx**2 + self.dy**2)) -
                 (self.dx**2 * self.dy**2) / (2 * (self.dx**2 + self.dy**2)) * 
                 b[1:-1, 1:-1])

            # Boundary conditions
            self.p[:, 0] = self.p[:, 1]     # inlet
            self.p[:, -1] = 0               # outlet
            self.p[0, :] = self.p[1, :]
            self.p[-1, :] = self.p[-2, :]
            
        return self.p


    def step(self):
        n_u = np.copy(self.u)
        n_v = np.copy(self.v)

        p = self.update_pressure()

        self.u[1:-1, 1:-1] = (n_u[1:-1, 1:-1] - 
                                n_u[1:-1, 1:-1] * (self.dt / self.dx) * (n_u[1:-1, 1:-1] - n_u[1:-1, 0:-2]) - 
                                n_v[1:-1, 1:-1] * (self.dt / self.dy) * (n_u[1:-1, 1:-1] - n_u[0:-2, 1:-1]) -
                                self.dt / (2 * self.rho * self.dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) + 
                                self.nu * (
                                    (self.dt / self.dx**2) * (n_u[1:-1, 2:] - 2 * n_u[1:-1, 1:-1] + n_u[1:-1, 0:-2]) + 
                                    (self.dt / self.dy**2) * (n_u[2:, 1:-1] - 2 * n_u[1:-1, 1:-1] + n_u[0:-2, 1:-1])))

        self.v[1:-1,1:-1] = (n_v[1:-1, 1:-1] -
                                n_u[1:-1, 1:-1] * (self.dt / self.dx) * (n_v[1:-1, 1:-1] - n_v[1:-1, 0:-2]) -
                                n_v[1:-1, 1:-1] * (self.dt / self.dy) * (n_v[1:-1, 1:-1] - n_v[0:-2, 1:-1]) -
                                self.dt / (2 * self.rho * self.dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                                self.nu * (
                                    (self.dt / self.dx**2) * (n_v[1:-1, 2:] - 2 * n_v[1:-1, 1:-1] + n_v[1:-1, 0:-2]) +
                                    (self.dt / self.dy**2) * (n_v[2:, 1:-1] - 2 * n_v[1:-1, 1:-1] + n_v[0:-2, 1:-1])))

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

        # Obstacle conditions
        self.u[self.obstacle] = 0
        self.v[self.obstacle] = 0

        self.time += self.dt
    
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

    def run_animation(self, steps=200, interval=50):
        """Run the simulation as an animation."""
        # Using a wider, shorter aspect ratio to prevent excessive whitespace
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # Calculate Reynolds number
        Re = (self.inlet * 2 * 0.05) / self.nu

        # Add an overall title with the Reynolds number
        fig.suptitle(
            f"Flow Past a Cylinder (Re ≈ {Re:.1f})", fontsize=16, fontweight="bold"
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

        # Add time counter
        step_text = fig.text(0.5, 0.92, f"Time: 0.000 s", ha="center", fontsize=12)

        def update(frame):
            for _ in range(
                100
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
            step_text.set_text(f"Time: {self.time:.3f} s")

            return vel_plot, vort_plot, step_text

        print(f"Starting animation with interval={interval} ms...")
        ani = animation.FuncAnimation(
            fig, update, frames=steps, interval=interval, blit=False, repeat=False
        )
        # Adjust layout to make room for the suptitle and horizontal colorbars
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.5)
        plt.show()
        print("Animation completed.")


model = finite_difference(resolution=0.01, rho=1.0, nu=0.001666, inlet=1)
print("Showing animation to demonstrate functionality...")
model.run_animation(steps=2000, interval=50)
