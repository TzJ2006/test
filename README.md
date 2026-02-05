# Cross-Platform CPU/GPU Benchmarking Tool

A unified benchmarking solution for measuring CPU and GPU FLOPS performance across different hardware platforms.

## Features

- **Cross-platform**: macOS, Linux, Windows
- **CPU Benchmarks**: Single-core and all-cores FLOPS measurement
- **GPU Support**: NVIDIA (CUDA), Apple Silicon (MPS), Intel (XPU), AMD (OpenCL)
- **Multiple Precisions**: FP64, FP32, FP16, BF16 (bfloat16), FP8 (experimental)
- **Accurate Measurement**: Warmup runs, statistical analysis (median, IQR outlier removal)
- **HTML Reports**: Interactive leaderboards and charts with historical data tracking

## Quick Start

### Installation

```bash
# Required dependencies
pip install torch numpy pandas plotly

# Optional dependencies
pip install threadpoolctl  # For precise BLAS thread control
pip install pyopencl       # For Intel/AMD GPU support
```

### Usage

```bash
# Run all benchmarks (results append to CSV)
python -m benchmark.cli

# CPU benchmarks only
python -m benchmark.cli --cpu-only

# GPU benchmarks only
python -m benchmark.cli --gpu-only

# Generate HTML report
python -m benchmark.cli --report-only

# Run benchmarks + auto-generate report
python -m benchmark.cli --report
```

### Output

- **CSV file**: `benchmark_results.csv` - All historical data (append mode)
- **HTML report**: `benchmark_report.html` - Interactive leaderboards and charts

## Benchmark Types

### CPU Benchmarks

| Benchmark | Description |
|-----------|-------------|
| Single-Core (Scalar) | Pure Python operations (sqrt + add) |
| Single-Core BLAS | NumPy matrix multiplication (1 thread) |
| All-Cores BLAS | NumPy matrix multiplication (all CPU cores) |

### GPU Benchmarks

| Precision | Description |
|-----------|-------------|
| FP64 | Double precision (64-bit) |
| FP32 | Single precision (32-bit) |
| FP16 | Half precision (16-bit) |
| BF16 | Brain Float 16 (better dynamic range) |
| FP8_exp | 8-bit float (experimental, limited support) |

## Historical Test Results

### GPU Speed Tests

### NVIDIA RTX 2060:
![2060](results/GPU%20speed%20test/2060.jpg)

### NVIDIA RTX 2080 Ti:
![2080Ti](results/GPU%20speed%20test/2080Ti.jpg)

### NVIDIA A100:
![A100](results/GPU%20speed%20test/A100.jpg)

### NVIDIA RTX 5000 Ada:
![RTX5000ada](results/GPU%20speed%20test/RTX5000ada.jpg)

### NVIDIA RTX 6000 Ada:
![RTX6000ada](results/GPU%20speed%20test/RTX6000ada.jpg)

### Apple M4 Pro:
![M4 pro](results/GPU%20speed%20test/M4%20pro.png)

## Module Structure

```
benchmark/
├── __init__.py    # Module exports
├── cli.py         # Command-line interface
├── core.py        # Base classes and utilities
├── cpu.py         # CPU benchmarks
├── detect.py      # Hardware detection
├── gpu.py         # GPU benchmarks
├── report.py      # HTML report generation
└── README.md      # Detailed documentation
```

## Advanced Usage

```bash
# Custom CSV output
python -m benchmark.cli --output my_results.csv

# Custom matrix size
python -m benchmark.cli --matrix-size 8192

# Generate report from specific CSV
python -m benchmark.cli --report-only --input-csv my_results.csv --report-output report.html

# Show system information
python -m benchmark.cli --info

# Quiet mode (minimal output)
python -m benchmark.cli --quiet
```

## CSV Format

The CSV file contains all benchmark results with the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | ISO format timestamp |
| `cpu_model` | CPU model name |
| `cpu_cores` | Number of CPU cores |
| `gpu_model` | GPU model name |
| `backend` | cuda/mps/xpu |
| `dtype` | FP64/FP32/FP16/BF16 |
| `flops_gflops` | Performance in GFLOPS |
| `os` | Operating system |
| `python_version` | Python version |
| `torch_version` | PyTorch version |
| `cuda_version` | CUDA version (if applicable) |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
