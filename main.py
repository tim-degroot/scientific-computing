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


# Benchmarking function
def benchmark_lbm(grid_sizes, num_steps=500, num_runs=5):
    """Benchmark LBM execution time for different grid sizes."""
    times = []
    ci_intervals = []
    for size in grid_sizes:
        run_times = []
        for _ in range(num_runs):
            lbm = D2Q9LBM(resolution=size, tau=0.6, u_inlet=0.1)
            start_time = time.time()
            for _ in range(num_steps):
                lbm.step()
            run_times.append(time.time() - start_time)
        avg_time = np.mean(run_times)
        times.append(avg_time)

        # Calculate 95% confidence interval
        confidence = 0.95
        n = len(run_times)
        h = sem(run_times) * t.ppf((1 + confidence) / 2, n - 1)
        ci_intervals.append(h)

    return times, ci_intervals

# Plotting function
def plot_benchmark(grid_sizes, results):
    """Plot benchmark results for multiple methods with confidence intervals."""
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        times, ci_intervals = data
        plt.errorbar(grid_sizes, times, yerr=ci_intervals, label=label, marker="o", linestyle="", capsize=5)
    plt.xlabel("Grid Size (NxN)")
    plt.ylabel("Average Execution Time (s)")
    plt.title("Execution Time vs Grid Size with 95% Confidence Intervals")
    plt.legend()
    plt.grid(True)
    plt.show()

# Stability analysis function
def analyze_stability(re_range, resolution, tau, u_inlet, num_steps):
    """Numerically analyze stability over a range of Reynolds numbers."""
    stability_results = {}
    for re in re_range:
        try:
            lbm = D2Q9LBM(resolution=resolution, tau=tau, u_inlet=u_inlet)
            max_rho, min_rho = 0, float('inf')
            max_u, max_v = 0, 0

            for step in range(num_steps):
                lbm.step()
                if step % 10 == 0:
                    lbm.visualize(step)  # Visualize every 10 steps
                max_rho = max(max_rho, np.max(lbm.rho))
                min_rho = min(min_rho, np.min(lbm.rho))
                max_u = max(max_u, np.max(np.abs(lbm.u)))
                max_v = max(max_v, np.max(np.abs(lbm.v)))

            stability_results[re] = {
                "max_rho": max_rho,
                "min_rho": min_rho,
                "max_u": max_u,
                "max_v": max_v,
            }
        except Exception as e:
            stability_results[re] = {
                "error": str(e)
            }
    return stability_results

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

    if "/" in PLOTS_DIR or "\\" in PLOTS_DIR:
        os.makedirs(os.path.dirname(PLOTS_DIR), exist_ok=True)

    if "/" in DATA_DIR or "\\" in DATA_DIR:
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

    # Benchmarking for different resolutions
    print("Running benchmarking tests...")
    resolutions = [0.02, 0.01, 0.005]  # Different resolutions in meters per grid point
    for resolution in resolutions:
        print(f"Benchmarking for resolution: {resolution} m/grid point")
        # Enclose resolution in a list so the benchmarking function iterates properly
        lbm_times, lbm_ci = benchmark_lbm([resolution]) 
        print(f"Resolution {resolution} m/grid point: Average execution times for LBM: {lbm_times}")

    # Numerical stability analysis
    print("Analyzing stability numerically...")
    re_range = range(10, 500, 10)
    stability_results = analyze_stability(re_range, resolution=0.01, tau=0.6, u_inlet=0.1, num_steps=500)
    for re, metrics in stability_results.items():
        if "error" in metrics:
            print(f"Re={re}: Unstable - {metrics['error']}")
        else:
            print(f"Re={re}: Stable")