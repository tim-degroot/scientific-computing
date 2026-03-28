from ngsolve import *
from netgen.occ import *
import numpy as np
import matplotlib.pyplot as plt

class helmholtz():
    def __init__(self, maxh=0.05, scale=3):
        
        self.A = 10e4
        self.omega = 0.2
        self.maxh = maxh
        
        f_wifi = 2.4e9 / scale
        c = 3e8         # Speed of light
        self.k0 = (2 * np.pi * f_wifi) / c
        
        
        self.mesh = None
        self.k_coeff = None
        

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


    def solve_helmholtz(self, router_pos):
        
        self.material_mapping()

        # 3. Bilinear Form
        # Finite Element Space (H1 elements)
        fes = H1(self.mesh, order=5, complex=True)
        u, v = fes.TnT()
        a = BilinearForm(fes)
        a += (grad(u)*grad(v) - self.k_coeff**2 * u * v) * dx

        # Boundary Condition: Impedance/Absorbing BC on the "outer" boundary
        a += -1j * self.k0 * u * v * ds("outer_wall")

        a.Assemble()

        self.router(router_pos)

        # Linear form
        f = LinearForm(self.pulse * v * dx).Assemble()

        # Solve
        self.gfu = GridFunction(fes, name="u")
        self.gfu.vec.data = a.mat.Inverse() * f.vec        
        
        self.db_signal = self.amplitude_to_db(self.gfu, self.A)

        # 3. Draw the result
        Draw(self.db_signal, self.mesh, "wifi_strength_dB", sd=4, autoscale=True)
                
        
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
            print(f"Room {i+1} at ({px}, {py}): Average Strength = {avg_strength_db:.4f}")

        print("-" * 30)
        print(f"TOTAL MEASURED SIGNAL STRENGTH: {total_signal_sum:.4f}")
        
        return total_signal_sum
                


def optimize_wifi_strength():
    
    sim = helmholtz(maxh=0.05)
    sim.floor_plan_mesh()
    
    router_positions = [(2.5, 5.5), (8, 6), (1, 1), (8, 9)]
    location_strength = {}
    
    for pos in router_positions:
        sim.solve_helmholtz(router_pos=pos)
        signal_sum = sim.wifi_strength()
        location_strength.update({pos:signal_sum})
        
    return location_strength

# if __name__=="__main__":

loc_strength = optimize_wifi_strength()
optimal_loc = max(loc_strength, key=loc_strength.get)

print(f"Optimal location: {optimal_loc} with strength {loc_strength[optimal_loc]}")