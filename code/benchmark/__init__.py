"""
Cross-platform CPU/GPU Benchmarking Tool

A unified benchmarking solution supporting:
- Platforms: macOS, Linux, Windows
- CPU: Single-core and all-core FLOPS
- GPU: NVIDIA (CUDA), Apple (MPS), Intel, AMD - all precision levels
"""

from .core import BaseBenchmark, fmt_flops, calculate_flops_gemm, RobustTimer, BenchmarkResults
from .detect import get_cpu_info, get_gpu_info, get_system_info
from .cpu import CpuSingleCoreBenchmark, CpuAllCoresBenchmark
from .gpu import GpuBenchmark
from .report import BenchmarkReport, generate_report

__all__ = [
    # Core
    'BaseBenchmark',
    'fmt_flops',
    'calculate_flops_gemm',
    'RobustTimer',
    'BenchmarkResults',
    # Detection
    'get_cpu_info',
    'get_gpu_info',
    'get_system_info',
    # CPU
    'CpuSingleCoreBenchmark',
    'CpuAllCoresBenchmark',
    # GPU
    'GpuBenchmark',
    # Report
    'BenchmarkReport',
    'generate_report',
]

__version__ = '1.1.0'
