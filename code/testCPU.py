#!/usr/bin/env python3
import argparse
import platform
import multiprocessing
import time
import math
import sys

def cpu_work(iterations: int) -> float:
    """
    A simple CPU‐bound task: sum of square‐roots.
    Returns the final sum so it can’t be optimized away.
    """
    s = 0.0
    for i in range(iterations):
        s += math.sqrt(i)
    return s

def run_single(iterations: int) -> float:
    t0 = time.perf_counter()
    cpu_work(iterations)
    return time.perf_counter() - t0

def run_multi(iterations: int, workers: int) -> float:
    # split iterations as evenly as possible
    chunk, rem = divmod(iterations, workers)
    tasks = [chunk + (1 if i < rem else 0) for i in range(workers)]
    t0 = time.perf_counter()
    with multiprocessing.Pool(processes=workers) as p:
        p.map(cpu_work, tasks)
    return time.perf_counter() - t0

def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform CPU speed tester (single- and multi-core)"
    )
    parser.add_argument(
        "-i","--iterations",
        type=int,
        default=1_000_000_000,
        help="Total number of loop iterations (default: 10 million)"
    )
    parser.add_argument(
        "-w","--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of processes for multi-core test (default: all cores)"
    )
    parser.add_argument(
        "--no-single",
        action="store_true",
        help="Skip single-core test"
    )
    parser.add_argument(
        "--no-multi",
        action="store_true",
        help="Skip multi-core test"
    )
    args = parser.parse_args()

    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {platform.python_implementation()} {platform.python_version()}")
    print(f"CPU cores (logical): {multiprocessing.cpu_count()}")
    print(f"Iterations: {args.iterations:,}")
    print()

    if not args.no_single:
        print("→ Single-core test")
        t_single = run_single(args.iterations)
        flops_single = args.iterations / t_single / 1_000_000_000
        print(f"   Time: {t_single:.3f} s   ({flops_single:,.2f} TFLOPS/s)")
        print()

    if not args.no_multi:
        workers = args.workers
        if workers < 2:
            print("Skipping multi-core (workers < 2).")
        else:
            print(f"→ Multi-core test ({workers} workers)")
            t_multi = run_multi(args.iterations, workers)
            flops_multi = workers * args.iterations / t_multi / 1_000_000_000
            print(f"   Time: {t_multi:.3f} s   ({flops_multi:,.2f} TFLOPS/s total)")
            if not args.no_single:
                speedup = (args.iterations / t_single) / (workers * args.iterations / t_multi)
                print(f"   Speedup vs single: {(t_single/t_multi):.2f}×")
        print()

if __name__ == "__main__":
    # Protect entry point for Windows multiprocessing
    main()
