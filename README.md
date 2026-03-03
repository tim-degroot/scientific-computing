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
    - Usage: `uv run main.py [--hide]
    - The `--hide` flag suppresses interactive plot windows for faster output generation.
- `MonteCarlo.py`: Contains the implementation of the Monte Carlo DLA in the `DLA` class.
- `GrayScott.py`: Contains the implementation of the Gray-Scott model in the `GrayScott` class.
- `pyproject.toml`: Defines the project dependencies and Python version.

## Generated Artifacts

### Figures

All figures are automatically generated and saved in the `/plots/` directory. Figures are also displayed via an interactive window during runtime unless the `--hide` argument is passed. 

Location: `/plots/`

- `MC_DLA_grid.png`: 2x2 grid showing final DLA states for different random seeds.
- `MC_DLA_matrix.png`:4x4 matrix exploring the parameter space of $p_s$ vs. seeds.
- `MC_DLA_ps_overlap.png`: Heatmap showing simulation overlap grouped by $p_s$.
- `MC_DLA_seed_overlap.png`: Heatmap showing simulation overlap grouped by random seed.
