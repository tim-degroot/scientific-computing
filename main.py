import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

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
        help="Hide interactive plot windows and only save files.",
    )
    # parser.add_argument("--load", action="store_true", help="Load previous simulations")
    args = parser.parse_args()

    show_plots = not args.hide
    # load_data = args.load

    if "/" in PLOTS_DIR or "\\" in PLOTS_DIR:
        os.makedirs(os.path.dirname(PLOTS_DIR), exist_ok=True)

    if "/" in DATA_DIR or "\\" in DATA_DIR:
        os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
