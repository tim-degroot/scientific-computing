import numpy as np
import os
import argparse
from lbm import D2Q9LBM
from finite_difference import finite_difference
from finite_element import NavierStokesSolver
import matplotlib.pyplot as plt
from helmholtz import Helmholtz

PLOTS_DIR = "plots/"
DATA_DIR = "data/"
PLOTS_FORMAT = ".pdf"

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
    
def push_Re(nu, tau):
    sim = NavierStokesSolver(nu=nu, tau=tau, tend=5.0)
    sim.make_mesh(maxh=0.04)
    sim.setup_spaces()
    sim.set_inflow()
    sim.assemble_system()
    sim.solve_initial_stokes()
    t, cd, cl = sim.run_simulation()
    
    # Plotting Results for Stability and Accuracy
    plt.figure(figsize=(12, 5))
    
    # Plot Lift (Accuracy Test)
    plt.subplot(1, 2, 1)
    plt.plot(t, cl, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("$C_L$")
    
    # Plot Drag (Stability Test)
    plt.subplot(1, 2, 2)
    plt.plot(t, cd, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("$C_D$")

    plt.tight_layout()
    plt.savefig(f"plots/FEM_Re_push.pdf")
    plt.show()

def reliability_test():
    
    # gridsizes to compare
    mesh_size = [0.1, 0.08, 0.06, 0.04]
    router_positions = [(5, 2), (5, 1), (2, 5), (8, 6), (1, 1), (8, 9)]
    
    mesh_loc_strength = {}
    for maxh in mesh_size:
        # set up mesh and solver
        sim = Helmholtz(maxh=maxh)
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
        
def plot_reliability_test(mesh_loc_strength, save_path=None, show=True):
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Scientific Computing Set 3 simulations."
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help="Hide the animation and only run the simulation in the background.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(PLOTS_DIR), exist_ok=True)
    os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)

    print("I. Navier-Stokes Equation")

    experiments_I1 = [
        {"resolution": 0.005, "Re": 100, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 250, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 500, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 750, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 1000, "u_inlet": 1, "steps": 10000},
        {"resolution": 0.005, "Re": 10000, "u_inlet": 1, "steps": 10000},
    ]

    experiments_I3 = [
        {"resolution": 0.005, "Re": 100, "u_inlet": 0.12, "steps": 10000},
        {"resolution": 0.005, "Re": 150, "u_inlet": 0.12, "steps": 10000},
        {"resolution": 0.005, "Re": 200, "u_inlet": 0.12, "steps": 15000},
        {"resolution": 0.005, "Re": 225, "u_inlet": 0.12, "steps": 15000},
        {"resolution": 0.005, "Re": 250, "u_inlet": 0.12, "steps": 15000},
    ]

    print("I.1 Finite Difference Method")
    for i, params in enumerate(experiments_I1):
        resolution, Re, u_inlet, steps = params.values()
        print(f"\nRunning experiment {i+1} with Re={Re}, u_inlet={u_inlet}")

        # Instantiate model
        fdm = finite_difference(Re, u_inlet, resolution)

        # Unified runner call
        fdm.run(steps=steps, animate=not args.hide, interval=5, steps_per_frame=10)
        print("   Simulation completed.")

        fdm.plot_histories(
            show=not args.hide,
            save_path=f"{PLOTS_DIR}lbm_r{resolution}_Re{Re:.0f}_u{u_inlet}_steps{steps}{PLOTS_FORMAT}",
        )
    
    print("I.2 Finite Elements Method (ngsolve)")
    
    print('Simulating mesh convergence')
    plot_mesh_convergence()
    
    print('Push FEM to highers Re')
    nu = 2.5e-4
    tau = 1e-4
    push_Re(nu, tau)

    print("I.3 Lattice Boltzmann Method")
    for i, params in enumerate(experiments_I3):
        resolution, Re, u_inlet, steps = params.values()
        print(f"\nRunning experiment {i+1} with Re={Re}, u_inlet={u_inlet}")
        Ma = u_inlet / (1 / np.sqrt(3))

        print(f"Re = {Re:.1f}, Ma = {Ma:.2f}")

        # Instantiate model
        lbm = D2Q9LBM(resolution, Re, u_inlet)

        # Unified runner call
        lbm.run(steps=steps, animate=not args.hide, interval=10, steps_per_frame=10)

        print("   Simulation completed.")

        lbm.plot_histories(
            show=not args.hide,
            save_path=f"{PLOTS_DIR}lbm_r{resolution}_Re{Re:.0f}_u{u_inlet}_steps{steps}{PLOTS_FORMAT}",
        )
        
    print("II. Optimial router location")
    
    print('Testing coarse mesh reliability')
    save_path=f"{PLOTS_DIR}reliability_test{PLOTS_FORMAT}"
    mesh_loc_strength = reliability_test()
    plot_reliability_test(mesh_loc_strength, save_path=save_path)
    
    
    save_path=f"{PLOTS_DIR}optimal_router_heat{PLOTS_FORMAT}"

    sim = Helmholtz(maxh=0.08)
    sim.floor_plan_mesh()
    xs, ys, Z, best_pos = sim.optimize_wifi_strength_grid(step=0.4)
    print("best position:", best_pos)
    sim.plot_strength_heatmap(xs, ys, Z, best_pos, save_path=save_path)

    # # Simulate best position on high res gridnet
    # sim = Helmholtz(maxh=0.03)
    # sim.floor_plan_mesh()
    # sim.setup_solver()
    # sim.solve_helmholtz((5.15, 3.55), draw=True)
    
    
        
        
