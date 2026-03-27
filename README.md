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
    - Usage: `uv run main.py [--hide]`
    - The `--hide` flag suppresses interactive plot windows for faster output generation. It also hides the live animation of the simulation which is not saved.
- `pyproject.toml`: Defines the project dependencies and Python version.

## Generated Artifacts

### Figures

All figures are automatically generated and saved in the `/plots/` directory. Figures are also displayed via an interactive window during runtime unless the `--hide` argument is passed. 

Location: `/plots/`

- `lbm_r0.005_Re100_u0.12_steps10000.pdf`: Physical Quantities over time within the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=100$, $u_\text{inlet}=0.12$, for 10,000 steps.
- `lbm_r0.005_Re150_u0.12_steps10000.pdf`: Physical Quantities over time within the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=150$, $u_\text{inlet}=0.12$, for 10,000 steps.
- `lbm_r0.005_Re200_u0.12_steps15000.pdf`: Physical Quantities over time within the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=200$, $u_\text{inlet}=0.12$, for 15,000 steps.
- `lbm_r0.005_Re225_u0.12_steps15000.pdf`: Physical Quantities over time within the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=225$, $u_\text{inlet}=0.12$, for 15,000 steps.
- `lbm_r0.005_Re250_u0.12_steps15000.pdf`: Physical Quantities over time within the Lattice Boltzmann Method simulation with $r=0.005$, $\text{Re}=250$, $u_\text{inlet}=0.12$, for 15,000 steps.
