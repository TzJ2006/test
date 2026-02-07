# Repository Guidelines

## Project Structure & Module Organization
- `benchmark/` contains the Python package:
  - `cli.py` entry point (`python -m benchmark.cli`)
  - `cpu.py`, `gpu.py` benchmark implementations
  - `detect.py` hardware/software detection
  - `core.py` shared timing and CSV persistence
  - `report.py` HTML report generation
- Root files:
  - `requirements.txt` dependency list
  - `benchmark_results.csv` appended benchmark history
  - `benchmark_report.html` generated visualization output
  - `results/` image assets for historical benchmark snapshots

## Build, Test, and Development Commands
- Install dependencies: `pip install -r requirements.txt`
- Run full benchmark: `python -m benchmark.cli`
- CPU only: `python -m benchmark.cli --cpu-only`
- GPU only: `python -m benchmark.cli --gpu-only`
- Generate report from CSV: `python -m benchmark.cli --report-only`
- Run benchmarks and refresh report: `python -m benchmark.cli --report`
- Show system info only: `python -m benchmark.cli --info`

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear docstrings for modules, classes, and public functions.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and UPPER_CASE for constants (for example `MATRIX_SIZE`).
- Keep benchmark result payloads as explicit dictionaries with stable keys (`dtype`, `backend`, `flops_per_sec`) to avoid CSV/report breakage.
- Prefer small, composable functions in `benchmark/` rather than embedding logic directly in the CLI path.

## Testing Guidelines
- No automated test suite is currently committed; validate changes with targeted CLI runs.
- Minimum validation before a PR:
  - `python -m benchmark.cli --cpu-only --duration 3`
  - `python -m benchmark.cli --gpu-only --duration 3` (if GPU is available)
  - `python -m benchmark.cli --report-only`
- If you add tests, place them under `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines
- Recent history uses short imperative commits (for example `update A6000`, `debug CPU`). Keep messages imperative and scoped.
- Recommended format: `<area>: <change>` (example: `gpu: improve FP16 timeout handling`).
- PRs should include:
  - concise summary of behavior changes
  - commands run for validation and key output notes
  - linked issue (if applicable)
  - updated report/screenshot when UI or benchmark output meaningfully changes
