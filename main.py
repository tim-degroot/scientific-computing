import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from lbm import D2Q9LBM
import time
from scipy.stats import sem, t

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

    print("Running LBM simulation...")
    lbm = D2Q9LBM(resolution=0.01, tau=0.6, u_inlet=0.2)

    if args.hide:
        print("Running simulation without animation...")
        for _ in range(1000):
            lbm.step()
        print("Simulation completed.")
    else:
        print("Showing animation to demonstrate functionality...")
        lbm.run_animation(steps=100, interval=100)
