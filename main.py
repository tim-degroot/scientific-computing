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
        # 1. High Mach Test: Original run with high velocity. 
        {"resolution": 0.005, "Re": 100, "u_inlet": 0.12, "steps": 10000},  # Re = 60.0, Ma = 0.35

        # # 2. Tweaking inlet velocity for better stability.
        # {"resolution": 0.01, "tau": 0.6, "u_inlet": 0.133, "steps": 2000},  # Re = 39.9, Ma = 0.23

        # # 3. Critical Stability Test: Original run with tau close to 0.5. 
        # {"resolution": 0.01, "tau": 0.55, "u_inlet": 0.1, "steps": 2000}, # Re = 60.0, Ma = 0.17

        # 4. Stable Shedding (Re = 144): Lower Ma for better stability.
        {"resolution": 0.005, "Re": 150, "u_inlet": 0.12, "steps": 10000}, # Re = 144.0, Ma = 0.10

        # 5. High Reynolds (Re = 400): Stronger vortex street, finer mesh.
        {"resolution": 0.003, "Re": 200, "u_inlet": 0.12, "steps": 10000}, # Re = 400.0, Ma = 0.10
    ]

    for i, params in enumerate(experiments):
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
            save_path=f"{PLOTS_DIR}lbm_{i+1}_r{resolution}_Re{Re:.0f}_u{u_inlet}_steps{steps}{PLOTS_FORMAT}",
        )
