# Scientific Computing: Assignment Set 2

## Quickstart

```bash
# Clone and enter the branch
git clone https://github.com/tim-degroot/scientific-computing.git
cd scientific-computing && git checkout set-2

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
    - The `--hide` flag suppresses interactive plot windows for faster output generation.
- `DLA.py`: Contains the implementation of the Diffusion-Limited Aggregation in the `DLA` class.
- `MonteCarlo.py`: Contains the implementation of the Monte Carlo DLA in the `MCDLA` class.
- `GrayScott.py`: Contains the implementation of the Gray-Scott model in the `GrayScott` class.
- `pyproject.toml`: Defines the project dependencies and Python version.

## Generated Artifacts

### Figures

All figures are automatically generated and saved in the `/plots/` directory. Figures are also displayed via an interactive window during runtime unless the `--hide` argument is passed. 

Location: `/plots/`

- `DLA_grid.pdf`: 2x2 grid showing final DLA states for different random seeds for DLA.
- `DLA_matrix.pdf` 4x4 grid exploring the parameter space of $\eta$ vs. seeds for DLA.
- `DLA_seed_overlap.pdf`: Heatmap showing simulation overlap grouped by random seed for DLA.
- `MC_DLA_grid.pdf`: 2x2 grid showing final DLA states for different random seeds for Monte Carlo DLA.
- `MC_DLA_matrix.pdf`: 4x4 grid exploring the parameter space of $p_s$ vs. seeds for Monte Carlo DLA.
- `MC_DLA_ps_overlap.pdf`: Heatmap showing simulation overlap grouped by $p_s$ for Monte Carlo DLA.
- `MC_DLA_seed_overlap.pdf`: Heatmap showing simulation overlap grouped by random seed for Monte Carlo DLA.
- `GrayScott.pdf`: 2x2 grid showing different patterns formed through reaction diffusion using the Gray Scott model.

### Data

The code outputs the computed results from DLA.DLA and MonteCarlo.DLA and stores them. After running the code one these values can be used to plot the results again using the `--load` argument.

Location: `/data/`

- `DLA_results.npy`: contains the array with dimensions (4, 4, 100, 100) containing 16 (100, 100) grids for combinations of four seeds and four $\eta$ values.
- `MC_DLA_results.npy`: contains the array with dimensions (4, 4, 100, 100) containing 16 (100, 100) grids for combinations of four seeds and four $p_s$ values.
