# AGENTS

This repository is a Python package implementing a production-grade CPT+ system optimized for Apple Silicon via MLX.

## Key information for coding agents

- Package root: `mlx_cpt_plus`
- Core modules:
  - `mlx_cpt_plus/core`: vocabulary, sequence storage, indexes, candidate management
  - `mlx_cpt_plus/tree`: prediction tree construction, traversal, compression
  - `mlx_cpt_plus/prediction`: retrieval, similarity scoring, voting, ranking, next-item prediction
  - `mlx_cpt_plus/mlx`: Apple Silicon MLX kernel wrappers and batch prediction support
  - `mlx_cpt_plus/compression`: subsequence compression and dictionary management
  - `mlx_cpt_plus/storage`: persistence/serialization support
  - `mlx_cpt_plus/online`: incremental updates and online learning
  - `mlx_cpt_plus/metrics`: evaluation metrics and model quality measures

## Recommended commands

- Run tests: `python -m pytest`
- Run a specific test file: `python -m pytest tests/path/to/test_file.py`
- Build/install package locally: `python -m pip install -e .`
- Dev dependencies are defined in `pyproject.toml` under `[project.optional-dependencies].dev`
- Example pipeline scripts:
  - `python scripts/build_index.py`
  - `python scripts/train_model.py`
  - `python scripts/evaluate_model.py`
  - `python scripts/export_model.py`

## Project conventions

- Python version: `>=3.9`
- Formatting:
  - `black` line length 100
  - `isort` profile `black`
- Static analysis:
  - `mypy` with `ignore_missing_imports = true`
  - `ruff` selected checks: `E, F, W, I, N, UP, B, C4`
- Tests live under `tests/` and use `pytest` naming conventions.

## Helpful files and documentation

- `README.md` — primary repository overview and usage instructions
- `CLAUDE.md` — architecture-oriented guidance for code assistants
- `pyproject.toml` — dependency, packaging, and tool configuration
- `.github/copilot-instructions.md` — chat customization for GitHub Copilot

## What to keep in mind

- This project is intended to stay pure Python while leveraging Apple Silicon MLX acceleration where appropriate.
- Preserve the separation of concerns between the model infrastructure, prediction logic, MLX kernel wrappers, and persistence layers.
- Avoid adding unrelated dependencies or non-Apple-specific hardware assumptions unless the repo explicitly supports them.
