import numpy as np
import os
import argparse
from lbm import D2Q9LBM

PLOTS_DIR = "plots/"
DATA_DIR = "data/"
PLOTS_FORMAT = ".pdf"

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
    print("I.1 Finite Difference Method")
    print("I.2 Finite Elements Method (ngsolve)")
    print("I.3 Lattice Boltzmann Method")

    experiments = [
        {"resolution": 0.01, "tau": 0.6, "u_inlet": 0.2},  # Re = 60.0, Ma = 0.35
        {"resolution": 0.01, "tau": 0.55, "u_inlet": 0.1}, # Re = 60.0, Ma = 0.17
        {"resolution": 0.01, "tau": 0.53, "u_inlet": 0.10},# Re = 100.0, Ma = 0.17
        {"resolution": 0.01, "tau": 0.52, "u_inlet": 0.08},# Re = 120.0, Ma = 0.14
        {"resolution": 0.01, "tau": 0.52, "u_inlet": 0.10},# Re = 150.0, Ma = 0.17
    ]

    for i, params in enumerate(experiments):
        resolution, tau, u_inlet = params.values()
        print(f"\nRunning experiment {i+1} with tau={tau}, u_inlet={u_inlet}")
        
        D_lattice = 2 * (0.05 / resolution) 
        nu_lattice = (tau - 0.5) / 3.0
        Re = (u_inlet * D_lattice) / nu_lattice
        Ma = u_inlet / (1/np.sqrt(3))
        
        print(f"Re = {Re:.1f}, Ma = {Ma:.2f}")
        
        # Instantiate model
        lbm = D2Q9LBM(resolution, tau, u_inlet)
        
        # Unified runner call
        if args.hide:
            lbm.run(steps=1000, animate=False, store_every=100)
        else:
            lbm.run(steps=1000, animate=True, interval=20)
            
        print("   Simulation completed.")
        
        lbm.plot_histories()