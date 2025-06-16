#!/usr/bin/env python3
"""
 testLaptop.py – quick CPU/GPU benchmark + optional stress test **with FLOPS logging**

 Usage
 -----
   python testLaptop.py                  # one-shot benchmark
   python testLaptop.py --stress 3       # 3-minute CPU+GPU stress + FLOPS report

 Notes
 -----
 • CPU FLOPS  = 2 * iterations (sqrt + add)
 • GPU FLOPS  = 2 * N³ * matmul_count  (FP32 GEMM)
 • Stress test upper-bounded to 10 min.
"""
import argparse, math, multiprocessing as mp, platform, time, threading, itertools

try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
CPU_ITERS   = 50_000_000          # scalar iter for quick bench
MATRIX_N    = 2048                # GEMM dim (FP32)
GEMM_ITERS  = 200                 # GPU reps

# ---------------------------------------------------------------------------

def fmt(x: float) -> str:
    for d, u in ((1e12, "TFLOPS"), (1e9, "GFLOPS"), (1e6, "MFLOPS")):
        if x >= d:
            return f"{x/d:,.2f} {u}/s"
    return f"{x:,.0f} FLOPS/s"

# =============================  QUICK BENCH  ==============================

def cpu_loop(n: int):
    s = 0.0
    for i in range(n):
        s += math.sqrt(i)


def quick_bench():
    print("\n=== Quick Laptop Benchmark ===")
    t0 = time.perf_counter(); cpu_loop(CPU_ITERS); dt = time.perf_counter() - t0
    print(f"CPU 1-core : {fmt(2*CPU_ITERS/dt)}   ({dt:.2f}s)")

    cores = mp.cpu_count()
    chunks = [CPU_ITERS//cores + (1 if i < CPU_ITERS % cores else 0) for i in range(cores)]
    t0 = time.perf_counter();
    with mp.Pool(cores) as p: p.map(cpu_loop, chunks)
    dt = time.perf_counter() - t0
    print(f"CPU {cores}-core: {fmt(2*CPU_ITERS/dt)}   ({dt:.2f}s)")

    if torch is None:
        print("PyTorch not installed – skip GPU part")
        return

    dev = (torch.device("cuda") if torch.cuda.is_available() else
           torch.device("mps")  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else None)
    if dev is None:
        print("No compatible GPU backend – skip GPU part"); return

    a = torch.randn(MATRIX_N, MATRIX_N, device=dev, dtype=torch.float32)
    b = torch.randn_like(a)
    _ = a @ b
    if dev.type == "cuda": torch.cuda.synchronize()
    elif dev.type == "mps": torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(GEMM_ITERS):
        _ = a @ b
    if dev.type == "cuda": torch.cuda.synchronize()
    elif dev.type == "mps": torch.mps.synchronize()
    dt = time.perf_counter() - t0
    flops = 2*MATRIX_N**3*GEMM_ITERS
    print(f"GPU {dev.type.upper():4}: {fmt(flops/dt)}   ({dt:.2f}s)")

# ===========================  STRESS + FLOPS  ============================

def stress_test(minutes: int):
    if minutes <= 0: return
    minutes = min(minutes, 10)
    duration = minutes*60
    end_t = time.time() + duration
    print(f"\n→ Stress for {minutes} min with FLOPS logging …")

    # ---- shared counters -------------------------------------------------
    cpu_counter = mp.Value('Q', 0)  # unsigned long long
    def cpu_spin(counter):
        local = 0
        while time.time() < end_t:
            for i in range(10000):
                local += 2  # 2 FLOPs / iter
            if local >= 100000:
                with counter.get_lock(): counter.value += local
                local = 0
    cores = mp.cpu_count(); pool = mp.Pool(cores, initializer=lambda c=cpu_counter: None)
    for _ in range(cores): pool.apply_async(cpu_spin, args=(cpu_counter,))

    # ---- GPU -------------------------------------------------------------
    gpu_flop = [0]
    dev = None
    if torch and (torch.cuda.is_available() or (getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())):
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
        N = 1024
        a = torch.randn(N, N, device=dev, dtype=torch.float16 if dev.type=="cuda" else torch.float32)
        b = torch.randn_like(a)
        per_mm = 2*N**3
        def gpu_spin():
            cnt = 0
            while time.time() < end_t:
                _ = a @ b
                cnt += 1
                if dev.type=="cuda": torch.cuda.synchronize()
                else: torch.mps.synchronize()
            gpu_flop[0] = cnt * per_mm
        gth = threading.Thread(target=gpu_spin); gth.start()
    else:
        gth = None

    # ---- wait ------------------------------------------------------------
    while time.time() < end_t:
        time.sleep(1)
    pool.terminate(); pool.join();
    if gth: gth.join()

    # ---- report ----------------------------------------------------------
    cpu_flops = cpu_counter.value
    print(f"CPU total: {fmt(cpu_flops/duration)} average over {minutes} min")
    if gth:
        print(f"GPU total: {fmt(gpu_flop[0]/duration)} average over {minutes} min")
    print("Stress test done.\n")

# ===============================  CLI  ====================================

def main():
    ap = argparse.ArgumentParser("Laptop quick benchmark + stress")
    ap.add_argument("--stress", type=int, default=0,
                    help="stress minutes (<=10) & report FLOPS")
    args = ap.parse_args()

    quick_bench()
    if args.stress:
        stress_test(args.stress)

if __name__ == "__main__":
    main()
