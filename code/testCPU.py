#!/usr/bin/env python3
import os
import time
import argparse
import platform
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits

def get_cpu_name():
    try:
        if platform.system() == 'Darwin':
            return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        elif platform.system() == 'Linux':
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        else:
            return subprocess.check_output([
                'powershell', '-Command',
                'Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty Name'
            ], text=True, stderr=subprocess.DEVNULL).splitlines()[0].strip()
    except:
        return platform.processor() or "Unknown CPU"

def measure_flops(matrix_size, repeats, num_threads):
    A = np.random.random((matrix_size, matrix_size))
    B = np.random.random((matrix_size, matrix_size))

    with threadpool_limits(limits=num_threads):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            np.dot(A, B)
            end = time.perf_counter()
            times.append(end - start)

    avg_time = sum(times) / len(times)
    ops = 2 * (matrix_size ** 3)
    gflops = ops / avg_time / 1e9
    print(f"Threads={num_threads} | Avg Time: {avg_time:.3f}s | GFLOPS: {gflops:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_size", type=int, default=8192, help="Size of matrix NxN")
    parser.add_argument("--cpu_iters", type=int, default=5, help="How many matrix multiplications to perform")
    args = parser.parse_args()

    logical_cores = os.cpu_count()
    print(f"\nPlatform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {platform.python_implementation()} {platform.python_version()}")
    print(f"Logical cores: {logical_cores}")
    print(f"CPU name: {get_cpu_name()}\n")

    print("🔹 Single-threaded (BLAS thread = 1)")
    measure_flops(args.matrix_size, args.cpu_iters, num_threads=1)

    print("\n🔹 Multi-threaded (BLAS thread = all logical cores)")
    measure_flops(args.matrix_size, args.cpu_iters, num_threads=logical_cores)