#!/usr/bin/env python3
import argparse
import platform
import time
import math
import multiprocessing

# ─── UTILs ───────────────────────────────────────────────────────────────────

def get_mps_device_name() -> str:
    import subprocess
    #  import re
    """
    On macOS, run `system_profiler SPDisplaysDataType` and grab the first
    'Chipset Model' line. Falls back to a generic name.
    """
    if platform.system() != "Darwin":
        return "Apple GPU (MPS)"
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Chipset Model:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Apple GPU (MPS)"

# ─── CPU TEST ────────────────────────────────────────────────────────────────

def cpu_work(iterations: int):
    s = 0.0
    for i in range(iterations):
        s += math.sqrt(i)  # 1 FLOP
        # + the add above is another FLOP


def run_cpu_single(iterations: int) -> float:
    t0 = time.perf_counter()
    cpu_work(iterations)
    return time.perf_counter() - t0


def run_cpu_multi(iterations: int, workers: int) -> float:
    chunk, rem = divmod(iterations, workers)
    sizes = [chunk + (1 if i < rem else 0) for i in range(workers)]
    t0 = time.perf_counter()
    with multiprocessing.Pool(workers) as pool:
        pool.map(cpu_work, sizes)
    return time.perf_counter() - t0


def format_gflops(flops: float) -> str:
    """Convert FLOPS to GFLOPS and format with two decimal places."""
    return f"{flops / 1e9:,.2f} GFLOPS/s"

# ─── PYTORCH GPU TEST (FP32, FP16, FP64, INT8) ──────────────────────────────

def run_pytorch_precision_gemm(size: int, iters: int, dev_name: str):
    try:
        import torch
    except ImportError:
        print("PyTorch not installed; skipping GPU tests.")
        return

    # Check backend availability
    if dev_name == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; skipping CUDA tests.")
        return
    if dev_name == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
        print("MPS unavailable; skipping MPS tests.")
        return

    device = torch.device(dev_name)

    if dev_name == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    elif dev_name == "mps":
        gpu_name = get_mps_device_name()
    else:
        gpu_name = str(device)
    print(f"\n→ PyTorch GEMM on {dev_name.upper()}: {size}×{size}, {iters} iters")
    print(f"   Device: {gpu_name}")

    # Test multiple precisions
    for dtype_name, dtype in [
        ("FP64", torch.float64),
        ("FP32", torch.float32),
        ("FP16", torch.float16),
        ("INT8", torch.int8)
    ]:
        # Prepare inputs
        try:
            if dtype_name == "INT8":
                a = torch.randint(-128, 127, (size, size), dtype=dtype, device=device)
                b = torch.randint(-128, 127, (size, size), dtype=dtype, device=device)
            else:
                a = torch.randn(size, size, dtype=dtype, device=device)
                b = torch.randn(size, size, dtype=dtype, device=device)

            # Warm-up
            c = a @ b
            if dev_name == "cuda":
                torch.cuda.synchronize()
            else:
                c.cpu()
        except Exception as e:
            print(f"   {dtype_name}: unsupported ({e}), skipping.")
            continue

        # Timed loop
        t0 = time.perf_counter()
        for _ in range(iters):
            if dtype_name != "INT8":
                a = torch.randn(size, size, dtype=dtype, device=device)
                b = torch.randn(size, size, dtype=dtype, device=device)
            # reuse a, b for INT8
            c = a @ b
        if dev_name == "cuda":
            torch.cuda.synchronize()
        else:
            c.cpu()
        dt = time.perf_counter() - t0

        # FLOPS count: 2 * N^3 * iters
        flops = 2 * size**3 * iters
        print(f"   {dtype_name}: Time: {dt:.3f}s   →  {format_gflops(flops / dt)}")

# ─── CORE ML / ANE TEST (macOS only) ─────────────────────────────────────────

def run_ane_test(dim: int, iters: int):
    if platform.system() != "Darwin":
        print("Not macOS; skipping ANE test.")
        return

    try:
        import coremltools as ct
        import numpy as np
        from coremltools.models import datatypes
        from coremltools.models.neural_network import NeuralNetworkBuilder
    except ImportError:
        print("coremltools (and numpy) not installed; skipping ANE.")
        return

    # Define model: y = Wx + b
    in_feat = [("input", datatypes.Array(dim))]
    out_feat = [("output", datatypes.Array(dim))]
    builder = NeuralNetworkBuilder(in_feat, out_feat)

    W = np.eye(dim, dtype=np.float32)
    b = np.zeros(dim, dtype=np.float32)

    builder.add_inner_product(
        name="dense",
        input_name="input",
        output_name="output",
        input_channels=dim,
        output_channels=dim,
        W=W,
        b=b,
        has_bias=True
    )

    spec = builder.spec

    for cu in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.ALL):
        print(f"\n→ Core ML inference on {cu.name}: dim={dim}, iters={iters}")
        mlmod = ct.models.MLModel(spec, compute_units=cu)
        x = {"input": np.random.rand(dim).astype(np.float32)}

        # Warm-up
        mlmod.predict(x)

        # Timed loop
        t0 = time.perf_counter()
        for _ in range(iters):
            mlmod.predict(x)
        dt = time.perf_counter() - t0

        # FLOPS count: 2 * dim^2 * iters
        flops = 2 * dim * dim * iters
        print(f"   Time: {dt:.3f}s   →  {format_gflops(flops / dt)}")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CPU + GPU + ANE benchmark tool")
    p.add_argument("--cpu-iters",    type=int, default=100_000_000,
                   help="CPU loop iterations")
    p.add_argument("--matrix-size",  type=int, default=4096,
                   help="GPU matmul dimension")
    p.add_argument("--matrix-iters", type=int, default=1000,
                   help="GPU matmul repetitions")
    p.add_argument("--ml-dim",       type=int, default=4096,
                   help="CoreML ANE inner product dim")
    p.add_argument("--ml-iters",     type=int, default=1000,
                   help="ANE inference repetitions")
    args = p.parse_args()

    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {platform.python_implementation()} {platform.python_version()}")
    num_cores = multiprocessing.cpu_count()
    print(f"Logical cores: {num_cores}")
    print()

    # CPU single-core
    print("→ CPU (single core)")
    t1 = run_cpu_single(args.cpu_iters)
    total_flops = 2 * args.cpu_iters
    print(f"   Time: {t1:.3f}s   →  {format_gflops(total_flops / t1)}")

    # CPU multi-core
    print(f"\n→ CPU (multi-core, {num_cores} workers)")
    t2 = run_cpu_multi(args.cpu_iters, num_cores)
    print(f"   Time: {t2:.3f}s   →  {format_gflops(total_flops / t2)}   Speedup: {t1/t2:.2f}×")

    # GPU precision GEMM
    run_pytorch_precision_gemm(args.matrix_size, args.matrix_iters, "cuda")
    run_pytorch_precision_gemm(args.matrix_size, args.matrix_iters, "mps")

    # ANE (Core ML)
    run_ane_test(args.ml_dim, args.ml_iters)

if __name__ == "__main__":
    main()
