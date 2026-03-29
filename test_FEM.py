import matplotlib.pyplot as plt
from finite_element import NavierStokesSolver
from ngsolve import *
from netgen.occ import *

def run_test(mesh_size, label):
    print(f"\n--- Testing {label} (maxh={mesh_size}) ---")
    sim = NavierStokesSolver(nu=0.001, tau=0.001, tend=5.0)
    sim.make_mesh(maxh=mesh_size)
    sim.setup_spaces()
    sim.set_inflow()
    sim.assemble_system()
    sim.solve_initial_stokes()
    return sim.run_simulation()

def plot_mesh_convergence():
    
    # Execute Tests
    t_coarse, cd_coarse, cl_coarse = run_test(0.08, "Coarse Mesh")
    t_medium, cd_medium, cl_medium = run_test(0.06, "Medium Mesh")
    t_fine, cd_fine, cl_fine = run_test(0.04, "Fine Mesh")

    # Plotting Results for Stability and Accuracy
    plt.figure(figsize=(12, 5))

    # Plot Lift (Accuracy Test)
    plt.subplot(1, 2, 1)
    plt.plot(t_coarse, cl_coarse, label="Coarse (maxh=0.08)", alpha=0.7)
    plt.plot(t_medium, cl_medium, label="Medium (maxh=0.06)", alpha=0.7)
    plt.plot(t_fine, cl_fine, label="Fine (maxh=0.04)", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("$C_L$")
    plt.legend()

    # Plot Drag (Stability Test)
    plt.subplot(1, 2, 2)
    plt.plot(t_coarse, cd_coarse, label="Coarse (maxh=0.08)", alpha=0.7)
    plt.plot(t_medium, cd_medium, label="Medium (maxh=0.06)", alpha=0.7)
    plt.plot(t_fine, cd_fine, label="Fine (maxh=0.04)", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("$C_D$")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/FEM_mesh_convergence.pdf")
    plt.show()
    
    
def push_Re():
    sim = NavierStokesSolver(nu=0.001, tau=0.001, tend=5.0)
    sim.make_mesh(maxh=0.04)
    sim.setup_spaces()
    sim.set_inflow()
    sim.assemble_system()
    sim.solve_initial_stokes()
    t, cd, cl = sim.run_simulation()
    
    # Plot Lift (Accuracy Test)
    plt.subplot(1, 2, 1)
    plt.plot(t, cl, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("$C_L$")
    plt.legend()
    
    # Plot Drag (Stability Test)
    plt.subplot(1, 2, 2)
    plt.plot(t, cd, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("$C_D$")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/FEM_Re_push.pdf")
    plt.show()
    
push_Re()