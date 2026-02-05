# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a cross-platform CPU/GPU benchmarking tool focused on measuring and comparing hardware performance. It uses a modular architecture with separate modules for CPU benchmarks, GPU benchmarks, hardware detection, and HTML report generation.

## Key Commands

### Running Benchmarks

```bash
# Run all benchmarks (results append to CSV)
python -m benchmark.cli

# CPU benchmarks only
python -m benchmark.cli --cpu-only

# GPU benchmarks only
python -m benchmark.cli --gpu-only

# Run without saving to CSV (for testing)
python -m benchmark.cli --no-save

# Show system information only
python -m benchmark.cli --info

# Generate HTML report from CSV
python -m benchmark.cli --report-only

# Run benchmarks + auto-generate report
python -m benchmark.cli --report

# Custom output paths
python -m benchmark.cli --output my_results.csv
python -m benchmark.cli --report-only --input-csv my_results.csv --report-output report.html

# Custom matrix size (affects BLAS and GPU benchmarks)
python -m benchmark.cli --matrix-size 8192

# Control benchmark duration per test (default: 10 seconds)
python -m benchmark.cli --duration 60  # 1 minute per benchmark
python -m benchmark.cli --duration 300  # 5 minutes per benchmark
```

### Default Parameters
- **Duration**: 10 seconds per benchmark (use `--duration` to change)
- CPU single-core: 10,000,000 iterations (scalar operations)
- CPU BLAS: matrix_size=8192 (single-core), 4096 (all-cores)
- GPU: matrix_size=2048, iterations=50

## Architecture

### Module Structure

```
benchmark/
├── __init__.py    # Module exports (BenchmarkReport, generate_report, etc.)
├── cli.py         # Command-line interface with argparse
├── core.py        # BaseBenchmark, RobustTimer, BenchmarkResults, CSV handling
├── cpu.py         # CpuSingleCoreBenchmark, CpuAllCoresBenchmark
├── detect.py      # get_cpu_info(), get_gpu_info(), get_system_info()
├── gpu.py         # GpuBenchmark (CUDA, MPS, XPU support)
├── report.py      # BenchmarkReport class (HTML generation with Plotly)
└── README.md      # Detailed module documentation
```

### Core Components

**core.py** - Base classes and utilities
- `BaseBenchmark` - Abstract base class for all benchmarks
- `RobustTimer` - Statistical timer with warmup and IQR outlier removal
- `BenchmarkResults` - CSV output manager (append mode)
- `calculate_flops_gemm()` - FLOPS calculation: `2 * N³`
- `calculate_flops_scalar()` - FLOPS calculation: `2 * iterations`

**cpu.py** - CPU benchmarks
- `CpuSingleCoreBenchmark` - Pure Python math.sqrt loop
- `CpuAllCoresBenchmark` - NumPy BLAS GEMM with all cores
- Uses `threadpoolctl` for precise thread control when available

**gpu.py** - GPU benchmarks
- `GpuBenchmark` - Unified GPU benchmark class
- Supported backends: CUDA (NVIDIA), MPS (Apple), XPU (Intel)
- Supported dtypes: FP64, FP32, FP16, BF16, FP8_exp (experimental)
- Automatic dtype detection per device

**detect.py** - Hardware detection
- `get_cpu_info()` - Cross-platform CPU model detection
- `get_gpu_info()` - GPU detection via PyTorch backends
- `get_system_info()` - Complete system info (CPU, GPU, software versions)

**report.py** - HTML report generation
- `BenchmarkReport` - Reads CSV, generates interactive HTML
- Uses Plotly for charts, Pandas for data processing
- Features: leaderboards, comparison charts, trend charts
- Self-growing: CSV accumulates data, report reflects all history

**cli.py** - Command-line interface
- Arguments: `--cpu-only`, `--gpu-only`, `--report`, `--report-only`
- Output options: `--output`, `--report-output`, `--input-csv`, `--no-save`
- Parameters: `--matrix-size`, `--iterations`, `--duration` (seconds per benchmark)
- Display modes: `--quiet`, `--verbose`, `--info`

## Implementation Notes

### Measurement Methodology
1. **Warmup Phase**: 5-100 iterations (depends on benchmark type)
2. **Measurement Phase**: 5-50 iterations
3. **Statistical Analysis**: Median time with IQR-based outlier removal
4. **GPU Synchronization**: Explicit `torch.cuda.synchronize()` or `torch.mps.synchronize()`

### FLOPS Calculation
- CPU scalar loop: `2 * iterations` (sqrt + add per iteration)
- GEMM (CPU/GPU): `2 * N³ * iterations` (multiply-add operations)

### Platform Support

| Platform | CPU | NVIDIA GPU | Apple GPU | Intel GPU | AMD GPU |
|----------|-----|------------|-----------|-----------|---------|
| Linux    | ✓   | ✓ (CUDA)   | ✗         | ✓ (XPU/OCL) | ✓ (OCL) |
| macOS    | ✓   | ✗          | ✓ (MPS)   | ✗         | ✗       |
| Windows  | ✓   | ✓ (CUDA)   | ✗         | ✓ (XPU/OCL) | ✓ (OCL) |

### dtype Support by Backend

| dtype   | CUDA | MPS  | XPU  | Notes                    |
|---------|------|------|------|--------------------------|
| FP64    | ✓    | ✗    | ✓    | MPS doesn't support FP64 |
| FP32    | ✓    | ✓    | ✓    | Universal                |
| FP16    | ✓    | ✓    | ✓    | Universal                |
| BF16    | ✓    | ✗    | ✓    | CUDA 8.0+, XPU          |
| FP8_exp | ✓*   | ✗    | ✗    | CUDA 8.9+, experimental  |

*FP8 matmul not fully supported in PyTorch yet

## CSV Data Flow

The CSV file uses **append mode** - it never overwrites existing data. This design enables:
1. **Accumulation**: Run benchmarks on different hardware, results append to the same file
2. **Historical Tracking**: All runs are preserved with timestamps
3. **Leaderboard Generation**: Report reads ALL data and shows best performance per hardware
4. **Self-Growing**: New hardware automatically appears when report is regenerated

**Important**: The `--output` option specifies which CSV to write to. If using a custom CSV for report generation, specify it with `--input-csv`.

## Dependencies

**Required:**
- `torch>=2.0.0` - GPU operations
- `numpy>=1.24.0` - CPU BLAS operations
- `pandas>=2.0.0` - CSV processing
- `plotly>=5.18.0` - HTML chart generation
- `tqdm>=4.65.0` - Progress bars

**Optional:**
- `threadpoolctl>=3.1.0` - Precise BLAS thread control
- `pyopencl>=2022.1` - Intel/AMD GPU support (fallback)

## Common Tasks

### Adding a New Benchmark Type

1. Create new class inheriting from `BaseBenchmark` in `core.py`
2. Implement `get_info()`, `run_iteration()`, and `get_flops()`
3. Add to appropriate module (`cpu.py`, `gpu.py`, or new file)
4. Update `cli.py` to include new benchmark

### Modifying Report Charts

Edit `report.py`:
- Chart generation methods: `create_*_chart()`
- HTML template: `HTML_TEMPLATE` (CSS styling)
- Add new figures to `self.figures` list in `generate_html()`

### Testing Changes

```bash
# Quick test (CPU only, small matrix, no CSV save, short duration)
python -m benchmark.cli --cpu-only --matrix-size 1024 --duration 1 --no-save

# Show system info without running benchmarks
python -m benchmark.cli --info

# Full test with report
python -m benchmark.cli --report

# Verify report opens in browser
# (Check benchmark_report.html)
```

## Notes

- GPU operations require proper synchronization for accurate timing
- Metal (MPS) has limited dtype support compared to CUDA
- FP8/INT8 matmul operations not fully supported in current PyTorch versions
- Results may vary between runs due to thermal throttling and background processes
- For best accuracy, close other applications and ensure consistent thermal state
