"""
GPU benchmarking - supports CUDA, MPS, XPU with all precision levels.
"""
import time
from typing import Dict, Any, List, Tuple, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .core import BaseBenchmark, calculate_flops_gemm


class GpuBenchmark:
    """
    Unified GPU benchmark supporting multiple backends and precision levels.

    Supported backends:
    - CUDA (NVIDIA)
    - MPS (Apple Silicon)
    - XPU (Intel oneAPI)

    Supported precision levels:
    - FP64, FP32, FP16, BF16 (bfloat16), FP8 (experimental, limited support)
    """

    # Default benchmark parameters
    MATRIX_SIZE = 2048
    ITERATIONS = 50

    # Data type configurations
    DTYPES = [
        ('FP64', 'float64'),
        ('FP32', 'float32'),
        ('FP16', 'float16'),
        ('BF16', 'bfloat16'),
    ]

    def __init__(self, matrix_size: int = None, iterations: int = None, duration: float = None):
        self.matrix_size = matrix_size or self.MATRIX_SIZE
        self.iterations = iterations or self.ITERATIONS
        self.duration = duration

        # Detect available backends
        self.backends = self._detect_backends()
        self.devices = self._get_all_devices()

    def _detect_backends(self) -> List[str]:
        """Detect available GPU backends."""
        backends = []

        try:
            import torch
            if torch.cuda.is_available():
                backends.append('cuda')
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                backends.append('mps')
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                backends.append('xpu')
        except ImportError:
            pass

        return backends

    def _get_all_devices(self) -> List[Dict[str, Any]]:
        """Get information about all available GPU devices."""
        devices = []

        try:
            import torch

            for backend in self.backends:
                if backend == 'cuda':
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        devices.append({
                            'backend': backend,
                            'index': i,
                            'name': props.name,
                            'memory_gb': props.total_memory / (1024 ** 3),
                            'compute_capability': f"{props.major}.{props.minor}",
                        })
                elif backend == 'mps':
                    devices.append({
                        'backend': backend,
                        'index': 0,
                        'name': 'Apple MPS',
                        'memory_gb': 0,
                        'compute_capability': 'MPS',
                    })
                elif backend == 'xpu':
                    for i in range(torch.xpu.device_count()):
                        props = torch.xpu.get_device_properties(i)
                        devices.append({
                            'backend': backend,
                            'index': i,
                            'name': getattr(props, 'name', f'Intel XPU {i}'),
                            'memory_gb': getattr(props, 'total_memory', 0) / (1024 ** 3),
                            'compute_capability': 'XPU',
                        })
        except ImportError:
            pass

        return devices

    def _get_supported_dtypes(self, device: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """
        Get supported data types for a device.

        Returns:
            List of (name, torch_dtype) tuples.
        """
        import torch

        supported = []
        backend = device['backend']

        # Base types supported by most backends
        base_dtypes = [
            ('FP32', torch.float32),
            ('FP16', torch.float16),
        ]

        # FP64 support varies by backend
        if backend == 'cuda':
            base_dtypes.insert(0, ('FP64', torch.float64))
        elif backend != 'mps':  # MPS doesn't support FP64
            base_dtypes.insert(0, ('FP64', torch.float64))

        # BF16 (bfloat16) - widely supported on modern GPUs
        if hasattr(torch, 'bfloat16'):
            if backend == 'cuda':
                # BF16 supported on CUDA with compute capability 8.0+
                cc = device.get('compute_capability', '0.0')
                major, _ = map(int, cc.split('.'))
                if major >= 8:
                    base_dtypes.append(('BF16', torch.bfloat16))
            elif backend == 'xpu':
                # Intel XPU typically supports BF16
                base_dtypes.append(('BF16', torch.bfloat16))
            # MPS has limited BF16 support, skip for now

        # FP8 support (CUDA only, compute capability 8.9+, experimental)
        # Note: FP8 matmul is not fully supported in most PyTorch versions
        # This is kept for future compatibility
        if backend == 'cuda' and hasattr(torch, 'float8_e4m3fn'):
            cc = device.get('compute_capability', '0.0')
            major, minor = map(int, cc.split('.'))
            if major > 8 or (major == 8 and minor >= 9):
                # Mark as experimental - may not work
                base_dtypes.append(('FP8_exp', torch.float8_e4m3fn))

        return base_dtypes

    def _create_tensors(self, n: int, dtype, device_obj):
        """Create random tensors for matrix multiplication."""
        import torch

        # Create tensors with appropriate initialization
        # Note: dtype.is_floating_point returns True for BF16
        if dtype.is_floating_point:
            try:
                a = torch.randn(n, n, dtype=dtype, device=device_obj)
                b = torch.randn(n, n, dtype=dtype, device=device_obj)
            except Exception:
                # Fallback to float32 then convert
                a = torch.randn(n, n, dtype=torch.float32, device=device_obj).to(dtype)
                b = torch.randn(n, n, dtype=torch.float32, device=device_obj).to(dtype)
        else:
            # For non-floating point types (like FP8 which is technically floating)
            # FP8 tensors need special handling
            try:
                a = torch.randn(n, n, dtype=dtype, device=device_obj)
                b = torch.randn(n, n, dtype=dtype, device=device_obj)
            except Exception:
                # Fallback: create float32 and convert
                a = torch.randn(n, n, dtype=torch.float32, device=device_obj)
                b = torch.randn(n, n, dtype=torch.float32, device=device_obj)
                try:
                    a = a.to(dtype)
                    b = b.to(dtype)
                except:
                    pass  # Keep as float32 if conversion fails

        return a, b

    def _synchronize(self, device_obj):
        """Synchronize device operations."""
        import torch

        device_type = device_obj.type
        if device_type == 'cuda':
            torch.cuda.synchronize(device_obj)
        elif device_type == 'mps':
            torch.mps.synchronize()
        elif device_type == 'xpu':
            torch.xpu.synchronize(device_obj)

    def _benchmark_device_dtype(self, device: Dict[str, Any],
                                 dtype_name: str, dtype,
                                 warmup_iters: int = 10, measure_iters: int = 50,
                                 show_progress: bool = True,
                                 progress_bar=None) -> Dict[str, Any]:
        """
        Benchmark a specific device and data type.

        Args:
            device: Device info dictionary
            dtype_name: Name of data type (e.g., 'FP32')
            dtype: PyTorch data type
            warmup_iters: Number of warmup iterations
            measure_iters: Number of measurement iterations (used if duration is None)
            show_progress: Whether to show progress (unused, kept for compatibility)
            progress_bar: Optional tqdm progress bar for updates

        Returns:
            Result dictionary with statistics.
        """
        import torch

        backend = device['backend']
        device_index = device['index']
        device_obj = torch.device(f"{backend}:{device_index}")

        # Create tensors (reuse to reduce overhead) - pre-generate before timing
        a, b = self._create_tensors(self.matrix_size, dtype, device_obj)

        # Test matmul support
        try:
            _ = a @ b
            self._synchronize(device_obj)
        except Exception as e:
            return {
                'name': f"{backend}:{device_index}",
                'type': 'gpu',
                'backend': backend,
                'dtype': dtype_name,
                'device_model': device['name'],
                'matrix_size': self.matrix_size,
                'error': str(e),
                'flops_per_sec': 0,
                'flops_formatted': 'N/A (not supported)',
            }

        # Warmup
        for _ in range(warmup_iters):
            _ = a @ b
            self._synchronize(device_obj)

        # Determine number of iterations
        if self.duration is not None:
            # Calibrate: run a few times to estimate single iteration time
            calib_times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = a @ b
                self._synchronize(device_obj)
                calib_times.append(time.perf_counter() - start)
            avg_time = sum(calib_times) / len(calib_times)
            measure_iters = max(1, int(self.duration / avg_time))

        # Measurement
        times = []
        for i in range(measure_iters):
            start = time.perf_counter()
            _ = a @ b
            self._synchronize(device_obj)
            times.append(time.perf_counter() - start)

            # Update progress bar if provided
            if progress_bar and measure_iters > 1:
                progress = int((i + 1) / measure_iters * 100)
                progress_bar.n = progress
                progress_bar.refresh()

        # Calculate median
        import numpy as np
        median_time = np.median(times)

        # Calculate FLOPS
        total_flops = calculate_flops_gemm(self.matrix_size, 1)
        flops_per_sec = total_flops / median_time if median_time > 0 else 0

        return {
            'name': f"{backend}:{device_index}",
            'type': 'gpu',
            'backend': backend,
            'dtype': dtype_name,
            'device_model': device['name'],
            'matrix_size': self.matrix_size,
            'iterations': measure_iters,
            'stats': {
                'median': median_time,
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
                'max': float(np.max(times)),
            },
            'flops_per_sec': flops_per_sec,
            'flops_formatted': self._format_flops(flops_per_sec),
        }

    def _format_flops(self, flops: float) -> str:
        """Format FLOPS to appropriate unit."""
        for unit, threshold in [('PFLOPS', 1e15), ('TFLOPS', 1e12),
                                 ('GFLOPS', 1e9), ('MFLOPS', 1e6)]:
            if flops >= threshold:
                return f"{flops / threshold:,.2f} {unit}/s"
        return f"{flops:,.2f} FLOPS/s"

    def run_all(self) -> List[Dict[str, Any]]:
        """
        Run benchmarks on all available devices with all supported precision levels.

        Returns:
            List of benchmark results.
        """
        if not self.backends:
            print("No GPU backends detected.\n")
            return []

        results = []
        total_benchmarks = sum(len(self._get_supported_dtypes(d)) for d in self.devices)

        print(f"Running GPU benchmarks...")
        print(f"  Detected {len(self.devices)} device(s) with {len(self.backends)} backend(s)\n")

        for device in self.devices:
            backend = device['backend']
            print(f"  [{device['name']}]")

            supported_dtypes = self._get_supported_dtypes(device)

            for dtype_name, dtype in supported_dtypes:
                # Use tqdm if available and duration is set
                if HAS_TQDM and self.duration and self.duration >= 5:
                    # For longer benchmarks, show progress
                    desc = f"    [{dtype_name}]"
                    with tqdm(total=100, desc=desc, unit='%', leave=False, ncols=80) as pbar:
                        result = self._benchmark_device_dtype(device, dtype_name, dtype,
                                                            show_progress=False, progress_bar=pbar)
                        pbar.update(100)  # Ensure it shows 100% complete
                        if 'error' in result:
                            tqdm.write(f"      ✗ ({result['error'][:30]}...)")
                        else:
                            tqdm.write(f"      ✓ {result['flops_formatted']}")
                else:
                    print(f"    {dtype_name}...", end=' ', flush=True)
                    result = self._benchmark_device_dtype(device, dtype_name, dtype)
                    if 'error' in result:
                        print(f"✗ ({result['error'][:30]}...)")
                    else:
                        print(f"✓ {result['flops_formatted']}")

                results.append(result)

        print("✓ GPU benchmarks complete.\n")
        return results


def run_all_gpu_benchmarks(matrix_size: int = None, iterations: int = None,
                           duration: float = None) -> list:
    """
    Run all GPU benchmarks and return results.

    Args:
        matrix_size: Size of matrices for GEMM benchmark
        iterations: Number of iterations (not used, kept for compatibility)
        duration: Target duration per benchmark in seconds

    Returns:
        List of benchmark result dictionaries.
    """
    bench = GpuBenchmark(matrix_size=matrix_size, iterations=iterations, duration=duration)
    return bench.run_all()
