from ngsolve import *
from netgen.occ import *
import numpy as np


def floor_plan_mesh():
    """
    inner_walls_coords: List of tuples ((x1, y1), (x2, y2)) for wall corners
    """
    # 1. Create the outer boundary (10x8)
    # We create the "air" first, then the thick outer walls
    outer_rect = Rectangle(10, 8).Face()
    inner_rect = Rectangle(9.7, 7.7).Face().Move(gp_Vec(0.15, 0.15, 0))
    
    # Outer walls are the difference
    outer_walls = outer_rect - inner_rect
    outer_walls.faces.name = "outer_wall"
    
    # 2. Create Inner Walls
    inner_walls_coords = [
    ((0, 2.925), (3, 3.075)),
    ((4, 2.925), (6, 3.075)),
    ((7, 2.925), (10, 3.075)),
    ((5.925, 3), (6.075, 8)),
    ((2.425, 0), (2.575, 2)),
    ((6.925, 2.5), (7.075, 3)),
    ((6.925, 0), (7.075, 1.5))]
    
    inner_walls_list = []
    for i, (p1, p2) in enumerate(inner_walls_coords):
        # Calculate width and height from coordinates
        w = abs(p2[0] - p1[0])
        h = abs(p2[1] - p1[1])
        wall = Rectangle(w, h).Face().Move(gp_Vec(p1[0], p1[1], 0))
        wall.faces.name = f"inner_wall_{i}"
        inner_walls_list.append(wall)
    
    # 3. Assemble the geometry
    # We subtract all inner walls from the 'inner_rect' (the air) 
    # so that the air domain doesn't overlap with the walls.
    air = inner_rect
    for wall in inner_walls_list:
        air = air - wall
    air.faces.name = "air"
    
    # Glue everything together into one geometry
    # This ensures a single mesh where boundaries are shared
    full_layout = Glue([air, outer_walls] + inner_walls_list)
    
    geo = OCCGeometry(full_layout, dim=2)
    
    # Generate mesh
    # maxh=0.1 ensures we have enough points to resolve the 0.15 thick walls
    mesh = Mesh(geo.GenerateMesh(maxh=0.1))
    return mesh

# Parameters
f_wifi = 2.4e9  # 2.4 GHz
c = 3e8         # Speed of light
k0 = 2 * np.pi * f_wifi / c

mesh = floor_plan_mesh()

# Finite Element Space (H1 elements)
fes = H1(mesh, order=5, complex=True)
u, v = fes.TnT()

# Gaussian pulse plus router position
omega = 0.2
router_pos = (4, 4)
pulse = 10e4*exp(-0.5*omega**2 * ((x-router_pos[0])**2 + (y-router_pos[1])**2))

# # Forms
# a = BilinearForm(fes)
# a += grad(u)*grad(v)*dx - omega**2*u*v*dx
# a += -omega*1j*u*v * ds("outer") # Add absorbing boundary condition on the outer "void"
# a.Assemble()

# 2. Define Material Mapping
# 'air' and the wall names must match the names defined in your floor_plan_mesh function
material_indices = {
    "air": 1.0,
    "outer_wall": 2.5 + 0.5j,
    "inner_wall": 2.5 + 0.5j
}

# Create the CoefficientFunction for n
n_coeff = CoefficientFunction([material_indices.get(mat, 1.0) for mat in mesh.GetMaterials()])
k_coeff = k0 * n_coeff

# 3. Bilinear Form
a = BilinearForm(fes)

# Internal Continuity: FEM handles this naturally at the interface 
# of different material domains as long as the mesh is conformal (Glued).
a += (grad(u)*grad(v) - k_coeff**2 * u * v) * dx

# Boundary Condition: Impedance/Absorbing BC on the "outer" boundary
# Note: Ensure your mesh boundary is named "outer" or change to "outer_boundary"
a += -1j * k0 * u * v * ds("outer_wall") 

a.Assemble()



f = LinearForm(pulse * v * dx).Assemble()


# Solve
gfu = GridFunction(fes, name="u")
gfu.vec.data = a.mat.Inverse() * f.vec
Draw(gfu, mesh, "wifi_strength")