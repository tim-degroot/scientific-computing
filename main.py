import numpy as np
import os
import argparse
from lbm import D2Q9LBM
from finite_difference import finite_difference

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
