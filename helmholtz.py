from ngsolve import *
from netgen.occ import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class helmholtz():
    def __init__(self, maxh=0.05, scale=1, order=5):
        
        self.A = 10e4
        self.omega = 0.2
        self.maxh = maxh
        self.order = order
        
        f_wifi = 2.4e9 / scale
        c = 3e8         # Speed of light
        self.k0 = (2 * np.pi * f_wifi) / c
        
        self.mesh = None
        self.k_coeff = None
        self.wall_rects = []
        self.bounds = (0.15, 9.85, 0.15, 7.85)
        self.measurement_points = [(1, 5), (2, 1), (9, 1), (9, 7)]
        self.no_router_radius = 0.5
        
    def floor_plan_mesh(self):
        # 1. Create the outer frame
        outer_rect = Rectangle(10, 8).Face()
        inner_rect = Rectangle(9.7, 7.7).Face().Move(gp_Vec(0.15, 0.15, 0))
        outer_walls = outer_rect - inner_rect
        outer_walls.faces.name = "outer_wall"
        
        # 2. Coordinates
        inner_walls_coords = [
            ((0, 2.925), (3, 3.075)),
            ((4, 2.925), (6, 3.075)),
            ((7, 2.925), (10, 3.075)),
            ((5.925, 3), (6.075, 8)),
            ((2.425, 0), (2.575, 2)),
            ((6.925, 2.5), (7.075, 3)),
            ((6.925, 0), (7.075, 1.5))
        ]
        
        inner_walls_list = []
        air = inner_rect
        self.wall_rects = []
        
        for i, (p1, p2) in enumerate(inner_walls_coords):
            w, h = abs(p2[0]-p1[0]), abs(p2[1]-p1[1])
            
            # Create the raw wall shape
            raw_wall = Rectangle(w, h).Face().Move(gp_Vec(p1[0], p1[1], 0))
            
            # INTERSECT with inner_rect so walls don't stick out into the outer frame
            # This keeps the geometry "clean"
            actual_wall = raw_wall * inner_rect 
            actual_wall.faces.name = f"inner_wall" # Grouping them under one name
            
            inner_walls_list.append(actual_wall)
            # Subtract from air to create the "hole"
            air = air - actual_wall
            
            self.wall_rects.append((min(p1[0], p2[0]), min(p1[1], p2[1]),
                                max(p1[0], p2[0]), max(p1[1], p2[1])))

        air.faces.name = "air"
        
        # 3. Assemble
        full_layout = Glue([air, outer_walls] + inner_walls_list)
        geo = OCCGeometry(full_layout, dim=2)
        
        # maxh note: at 2.4GHz, k~50. To resolve waves inside walls (n=2.5),
        # you need maxh around 0.02 to 0.05 for high accuracy.
        self.mesh = Mesh(geo.GenerateMesh(maxh=self.maxh))
        
    def material_mapping(self):
        
        # 2. Define Material Mapping
        # 'air' and the wall names must match the names defined in your floor_plan_mesh function
        n_air = 1.0
        n_wall = 2.5 + 0.5j

        # Map materials to their respective k-values (k = k0 * n)
        # Make sure these strings match your mesh.GetMaterials()
        material_map = {
            "air": self.k0 * n_air,
            "outer_wall": self.k0 * n_wall,
            "inner_wall": self.k0 * n_wall
        }

        # Create a CoefficientFunction for k over the whole domain
        self.k_coeff = CoefficientFunction([material_map.get(mat, self.k0) for mat in self.mesh.GetMaterials()])

    def router(self, pos):
        
        # Gaussian pulse from router position
        self.pulse = self.A*exp(-0.5* (self.omega**2) * ((x-pos[0])**2 + (y-pos[1])**2))

    def amplitude_to_db(self, gfu, ref=1):
        """
        gfu: The GridFunction (complex) from your Helmholtz solver
        reference_A: The amplitude that represents 0 dB
        """
        # 1. Get the magnitude squared: |u|^2 = real^2 + imag^2
        mag_sq = gfu.real**2 + gfu.imag**2
        
        # 2. Define a tiny epsilon to prevent log(0)
        eps = 1e-12
        
        # 3. Calculate 10 * log10(|u|^2 / A^2)
        # This is mathematically identical to 20 * log10(|u| / A)
        ratio = mag_sq / (ref**2)
        
        # We use IfPos to safely handle areas where the signal is zero
        return 10 * log(IfPos(ratio - eps, ratio, eps)) / log(10)
                
    def setup_solver(self):
        
        self.material_mapping()
        
        # Finite Element Space (H1 elements)
        self.fes = H1(self.mesh, order=self.order, complex=True)
        self.u, self.v = self.fes.TnT()
        self.a = BilinearForm(self.fes)
        self.a += (grad(self.u)*grad(self.v) - self.k_coeff**2 * self.u * self.v) * dx
        self.a += -1j * self.k0 * self.u * self.v * ds("outer_wall")
        self.a.Assemble()

        self.invA = self.a.mat.Inverse(self.fes.FreeDofs())
        self.gfu = GridFunction(self.fes, name="u")

    def solve_helmholtz(self, router_pos):
        
        self.router(router_pos)
        
        f = LinearForm(self.fes)
        f += self.pulse * self.v * dx
        f.Assemble()
        
        self.gfu.vec.data = self.invA * f.vec
        self.db_signal = self.amplitude_to_db(self.gfu, self.A)
        
        # Draw only if needed (-m netgen)
        Draw(self.db_signal, self.mesh, "wifi_strength_dB", autoscale=True)
                        
    def wifi_strength(self, rad=0.5):
        
        # Measurement points (Living room, Kitchen, Bathroom, Bedroom)
        points = [(1, 5), (2, 1), (9, 1), (9, 7)]
        
        total_signal_sum = 0
        signal_magnitude = self.db_signal # Standard magnitude for wifi strength

        print("--- Signal Strength Report ---")

        for i, (px, py) in enumerate(points):
            
            # Create a mask for the circular region around the point
            dist_sq = (x - px)**2 + (y - py)**2
            
            # If distance < radius, value is 1, else 0
            indicator = IfPos(rad**2 - dist_sq, 1, 0)
            
            # Calculate the integral of |u| over that specific circle
            integral_val = Integrate(signal_magnitude * indicator, self.mesh)
            
            # Calculate the area (denominator)
            # While mathematically pi*r^2, integrating the indicator is more numerically robust
            area = Integrate(indicator, self.mesh)
            
            if area > 0:
                avg_strength_db = np.real(integral_val / area)
                
            else:
                avg_strength_db = 0
                print(f"Warning: Point {px, py} is outside the mesh or circle is too small for maxh!")

            total_signal_sum += avg_strength_db
            print(f"Room {i+1} at ({px}, {py}): Average Strength = {avg_strength_db:.4f} dB")

        print("-" * 30)
        print(f"TOTAL MEASURED SIGNAL STRENGTH: {total_signal_sum:.4f} dB")
        
        return total_signal_sum
              
    def position_inside_air(self, pos):
        x, y = pos
        xmin, xmax, ymin, ymax = self.bounds
        if x <= xmin or x >= xmax or y <= ymin or y >= ymax:
            return False
        for x1, y1, x2, y2 in self.wall_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return False
        return True

    def router_position_allowed(self, pos):
            if not self.position_inside_air(pos):
                return False
            x, y = pos
            for px, py in self.measurement_points:
                if (x - px)**2 + (y - py)**2 < self.no_router_radius**2:
                    return False
            return True

    def generate_router_positions(self, step):
        xmin, xmax, ymin, ymax = self.bounds
        xs = np.arange(xmin + step/2, xmax, step)
        ys = np.arange(ymin + step/2, ymax, step)
        positions = [(x, y) for y in ys for x in xs]
        return positions, xs, ys

    def optimize_wifi_strength_grid(self, step=0.5):
        
        if self.mesh is None:
            self.floor_plan_mesh()
            
        self.setup_solver()
        _, xs, ys = self.generate_router_positions(step)
        Z = np.full((len(ys), len(xs)), np.nan)
        best_strength = -np.inf
        best_pos = None

        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                # Check if router is near wall or measure equipment
                if not self.router_position_allowed((x, y)):
                    continue
                
                # Solve for router position
                self.solve_helmholtz((x, y))
                strength = self.wifi_strength()
                Z[iy, ix] = strength
                
                # Check best position
                if strength > best_strength:
                    best_strength = strength
                    best_pos = (x, y)

        return xs, ys, Z, best_pos

    def plot_floorplan_overlay(self):
        ax = plt.gca()
        outer = patches.Rectangle((0, 0), 10, 8, fill=False, edgecolor="black", lw=2)
        ax.add_patch(outer)
        for x1, y1, x2, y2 in self.wall_rects:
            wall = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, edgecolor="black", lw=1.2)
            ax.add_patch(wall)

    def plot_strength_heatmap(self, xs, ys, Z, title="Router strength heatmap"):
        X, Y = np.meshgrid(xs, ys)
        cmap = plt.cm.inferno
        cmap.set_bad(color="lightgray")
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(X, Y, Z, shading="auto", cmap=cmap)
        plt.colorbar(label="Total wifi strength (dB)")
        self.plot_floorplan_overlay()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(title)
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.show()

    def reliability_test():
        
        # gridsizes to compare
        mesh_size = [0.1, 0.08, 0.06, 0.04]
        router_positions = [(2.5, 5.5), (8, 6), (1, 1), (8, 9), (5, 2), (5, 1)]
        
        mesh_loc_strength = {}
        for maxh in mesh_size:
            # set up mesh and solver
            sim = helmholtz(maxh=maxh)
            sim.floor_plan_mesh()
            sim.setup_solver()
            location_strength = {}
            
            # Solve for different router positions
            for pos in router_positions:
                sim.solve_helmholtz(router_pos=pos)
                signal_sum = sim.wifi_strength()
                location_strength.update({pos:signal_sum})
            
            mesh_loc_strength.update({maxh:location_strength})
            
        return mesh_loc_strength
            
    def plot_reliability_test(mesh_loc_strength):
        if not mesh_loc_strength:
            return

        positions = list(next(iter(mesh_loc_strength.values())).keys())
        x = np.arange(len(positions))
        labels = [f"{p[0]:.1f},{p[1]:.1f}" for p in positions]

        plt.figure()
        for maxh, location_strength in sorted(mesh_loc_strength.items()):
            strengths = [location_strength[pos] for pos in positions]
            plt.plot(x, strengths, marker="o", label=f"maxh={maxh}")

        plt.xticks(x, labels, rotation=45)
        plt.xlabel("Router position (x, y)")
        plt.ylabel("Total wifi strength (dB)")
        plt.title("Reliability of coarse mesh for optimal router location")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="mesh size")
        plt.tight_layout()
        plt.show()




# if __name__=="__main__":

# mesh_loc_strength = reliability_test()
# plot_reliability_test(mesh_loc_strength)

sim = helmholtz(maxh=0.08)
sim.floor_plan_mesh()
xs, ys, Z, best_pos = sim.optimize_wifi_strength_grid(step=0.5)
print("best position:", best_pos)
sim.plot_strength_heatmap(xs, ys, Z)