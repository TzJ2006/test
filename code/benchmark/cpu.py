"""
CPU benchmarking - single-core and all-cores FLOPS measurement.
"""
import math
import multiprocessing
import numpy as np
from typing import Dict, Any

try:
    from threadpoolctl import threadpool_limits
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False

from .core import BaseBenchmark, calculate_flops_scalar, calculate_flops_gemm


# Scalar loop for single-core benchmark
def _cpu_loop(n: int) -> float:
    """Pure Python CPU loop (sqrt + add)."""
    s = 0.0
    for i in range(n):
        s += math.sqrt(i)
    return s


class CpuSingleCoreBenchmark(BaseBenchmark):
    """
    Single-core CPU benchmark using pure Python operations.

    This measures CPU performance without leveraging BLAS or vectorization.
    """

    # Number of operations per iteration (reduced for faster testing)
    ITERATIONS = 10_000_000

    def __init__(self, warmup_iters: int = 5, measure_iters: int = 5):
        # Reduce iterations for single-core since it's slower
        super().__init__(warmup_iters, measure_iters)
        self.iterations = self.ITERATIONS

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'CPU Single-Core',
            'type': 'cpu_single_core',
            'backend': 'cpu',
            'dtype': 'N/A',
            'iterations': self.iterations,
        }

    def run_iteration(self) -> None:
        _cpu_loop(self.iterations)

    def get_flops(self, iterations: int) -> int:
        return calculate_flops_scalar(self.iterations * iterations)


class CpuAllCoresBenchmark(BaseBenchmark):
    """
    All-cores CPU benchmark using NumPy BLAS GEMM.

    This leverages optimized BLAS libraries (OpenBLAS, MKL, Accelerate)
    for multi-threaded matrix multiplication.
    """

    # Default matrix size for GEMM
    MATRIX_SIZE = 4096

    def __init__(self, matrix_size: int = None, num_threads: int = None,
                 warmup_iters: int = 3, measure_iters: int = 5):
        super().__init__(warmup_iters, measure_iters)
        self.matrix_size = matrix_size or self.MATRIX_SIZE
        self.num_threads = num_threads or multiprocessing.cpu_count()

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': f'CPU All-Cores ({self.num_threads} threads)',
            'type': 'cpu_all_cores',
            'backend': 'cpu',
            'dtype': 'float32',
            'matrix_size': self.matrix_size,
            'iterations': self.timer.measure_iters,
        }

    def run_iteration(self) -> None:
        # Generate matrices
        A = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)
        B = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)

        # Limit BLAS threads if threadpoolctl is available
        if HAS_THREADPOOLCTL:
            with threadpool_limits(limits=self.num_threads):
                np.dot(A, B)
        else:
            np.dot(A, B)

    def get_flops(self, iterations: int) -> int:
        return calculate_flops_gemm(self.matrix_size, iterations)


class CpuSingleCoreBLASBenchmark(BaseBenchmark):
    """
    Single-core CPU benchmark using NumPy BLAS with thread limit.

    This measures single-core BLAS performance by limiting NumPy to 1 thread.
    """

    MATRIX_SIZE = 2048

    def __init__(self, matrix_size: int = None, warmup_iters: int = 3,
                 measure_iters: int = 5):
        super().__init__(warmup_iters, measure_iters)
        self.matrix_size = matrix_size or self.MATRIX_SIZE

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'CPU Single-Core BLAS',
            'type': 'cpu_single_core_blas',
            'backend': 'cpu',
            'dtype': 'float32',
            'matrix_size': self.matrix_size,
            'iterations': self.timer.measure_iters,
        }

    def run_iteration(self) -> None:
        # Generate matrices
        A = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)
        B = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)

        # Limit to 1 BLAS thread
        if HAS_THREADPOOLCTL:
            with threadpool_limits(limits=1):
                np.dot(A, B)
        else:
            np.dot(A, B)

    def get_flops(self, iterations: int) -> int:
        return calculate_flops_gemm(self.matrix_size, iterations)


def run_all_cpu_benchmarks() -> list:
    """
    Run all CPU benchmarks and return results.

    Returns:
        List of benchmark result dictionaries.
    """
    results = []

    print("Running CPU benchmarks...")

    # Single-core pure Python
    print("  [1/3] Single-core (scalar operations)...")
    bench1 = CpuSingleCoreBenchmark()
    result1 = bench1.benchmark()
    results.append(result1)
    print(f"       Result: {result1['flops_formatted']}")

    # Single-core BLAS
    print("  [2/3] Single-core BLAS (matrix multiplication)...")
    bench2 = CpuSingleCoreBLASBenchmark()
    result2 = bench2.benchmark()
    results.append(result2)
    print(f"       Result: {result2['flops_formatted']}")

    # All-cores BLAS
    print("  [3/3] All-cores BLAS (matrix multiplication)...")
    bench3 = CpuAllCoresBenchmark()
    result3 = bench3.benchmark()
    results.append(result3)
    print(f"       Result: {result3['flops_formatted']}")

    print("✓ CPU benchmarks complete.\n")

    return results
