# Scientific Computing: Assignment Set 3

## Quickstart

```bash
# Clone and enter the branch
git clone https://github.com/tim-degroot/scientific-computing.git
cd scientific-computing && git checkout set-3

# Run with interactive plots (pop-up windows)
uv run main.py

# Run headlessly (save files only)
uv run main.py --hide
```

## Requirements & Usage

- **Python:** 3.10+
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **Dependencies**: Managed via `pyproject.toml` (includes `numpy`, `matplotlib`, `scipy` and `numba`)

`uv run` can be used to run the code for this assignment:

```bash
uv run main.py
```

Alternatively, you can use `uv sync` to manually update the environment in `.venv` and activate it or use `pip install .`

## Project Structure

- `main.py`: The entry point. Handles simulation loops, parameter configuration, and plotting output.
    - Imports and runs the finite element, finite difference, lattice Boltzmann, and Helmholtz simulations.
    - Usage: `uv run main.py [--hide]`
    - The `--hide` flag suppresses interactive plot windows and animation during simulation.
- `finite_element.py`: Contains `NavierStokesSolver`, which builds the 2D mesh, defines the mixed velocity-pressure finite element space, assembles the Stokes/implicit Euler matrices, and computes drag/lift on the cylinder.
- `finite_difference.py`: Implements the finite difference solver used by the FDM experiments in `main.py`.
- `lbm.py`: Implements the `D2Q9LBM` lattice Boltzmann model used by the LBM experiments in `main.py`.
- `helmholtz.py`: Implements the `Helmholtz` solver used for the WiFi/router placement reliability test in `main.py`.
- `pyproject.toml`: Defines the Python version and package dependencies required to run `main.py`.
- `data/`: Runtime data directory created by `main.py`. It is currently prepared for any saved data output.
- `plots/`: Output directory where `main.py` saves generated figures and result plots.

## Running `main.py`

1. From the repository root, install or sync the environment:

```bash
uv sync
```

2. Run the full simulation with interactive plots:

```bash
uv run main.py
```

3. Run headlessly and save figures only:

```bash
uv run main.py --hide
```

4. If you prefer a standard environment, install the package locally:

```bash
pip install .
```

Then run:

```bash
python main.py
```

## Generated Artifacts

### Figures

All figures are automatically generated and saved in the `/plots/` directory. Figures are also displayed via an interactive window during runtime unless the `--hide` argument is passed.

Location: `/plots/`

- `lbm_r0.005_Re100_u1_steps10000.pdf`: Physical Quantities over time for the Finite Difference Method simulation with $r=0.005$, $\text{Re}=100$, $u_\text{inlet}=1.0$, for 10,000 steps.
- `lbm_r0.005_Re250_u1_steps10000.pdf`: Physical Quantities over time for the Finite Difference Method simulation with $r=0.005$, $\text{Re}=250$, $u_\text{inlet}=1.0$, for 10,000 steps.
- `lbm_r0.005_Re500_u1_steps10000.pdf`: Physical Quantities over time for the Finite Difference Method simulation with $r=0.005$, $\text{Re}=500$, $u_\text{inlet}=1.0$, for 10,000 steps.
- `lbm_r0.005_Re750_u1_steps10000.pdf`: Physical Quantities over time for the Finite Difference Method simulation with $r=0.005$, $\text{Re}=750$, $u_\text{inlet}=1.0$, for 10,000 steps.
- `lbm_r0.005_Re1000_u1_steps10000.pdf`: Physical Quantities over time for the Finite Difference Method simulation with $r=0.005$, $\text{Re}=1000$, $u_\text{inlet}=1.0$, for 10,000 steps.
- `lbm_r0.005_Re10000_u1_steps10000.pdf`: Physical Quantities over time for the Finite Difference Method simulation with $r=0.005$, $\text{Re}=10000$, $u_\text{inlet}=1.0$, for 10,000 steps.
- `lbm_r0.005_Re100_u0.12_steps10000.pdf`: Physical Quantities over time for the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=100$, $u_\text{inlet}=0.12$, for 10,000 steps.
- `lbm_r0.005_Re150_u0.12_steps10000.pdf`: Physical Quantities over time for the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=150$, $u_\text{inlet}=0.12$, for 10,000 steps.
- `lbm_r0.005_Re200_u0.12_steps15000.pdf`: Physical Quantities over time for the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=200$, $u_\text{inlet}=0.12$, for 15,000 steps.
- `lbm_r0.005_Re225_u0.12_steps15000.pdf`: Physical Quantities over time for the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=225$, $u_\text{inlet}=0.12$, for 15,000 steps.
- `lbm_r0.005_Re250_u0.12_steps15000.pdf`: Physical Quantities over time for the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=250$, $u_\text{inlet}=0.12$, for 15,000 steps.
- `FEM_mesh_convergence.pdf`: Comparison of FEM drag and lift over time for coarse, medium, and fine meshes.
- `FEM_Re_push.pdf`: FEM drag and lift over time for the pushed Reynolds-number stability test.
- `reliability_test.pdf`: Total WiFi strength versus router position for different mesh sizes in the Helmholtz reliability test.
- `optimal_router_heat.pdf`: WiFi strength heatmap showing the best router location in the floor-plan optimization.
