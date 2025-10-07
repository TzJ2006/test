#!/usr/bin/env python3
"""
benchmark.py — CPU / GPU / Tensor-Core / Multi-GPU FLOPS 基准
2025-06 • MIT License
"""
import argparse, platform, time, math, multiprocessing, threading, subprocess
from tqdm import trange

# ─── UTILITIES ───────────────────────────────────────────────────────────────
def get_mps_name() -> str:
    if platform.system() != "Darwin": return "Apple GPU (MPS)"
    try:
        out = subprocess.check_output(["system_profiler","SPDisplaysDataType"],
                                      text=True, stderr=subprocess.DEVNULL)
        for ln in out.splitlines():
            if ln.strip().startswith("Chipset Model:"):
                return ln.split(":",1)[1].strip()
    except Exception: pass
    return "Apple GPU (MPS)"

def fmt(v: float) -> str:
    for d,u in [(1e15,"PFLOPS"),(1e12,"TFLOPS"),(1e9,"GFLOPS"),
                (1e6,"MFLOPS"),(1e3,"KFLOPS")]:
        if v>=d: return f"{v/d:,.2f} {u}/s"
    return f"{v:,.2f} FLOPS/s"

# ─── CPU BENCHMARK ───────────────────────────────────────────────────────────
def cpu_loop(n:int):
    s=0.0
    for i in range(n):
        s+=math.sqrt(i)

def bench_cpu_single(n:int)->float:
    t=time.perf_counter(); cpu_loop(n); return time.perf_counter()-t

def bench_cpu_multi(n:int,w:int)->float:
    seg=[n//w+(i<n%w) for i in range(w)]
    t=time.perf_counter()
    with multiprocessing.Pool(w) as p: p.map(cpu_loop,seg)
    return time.perf_counter()-t

# ─── TORCH HELPERS ───────────────────────────────────────────────────────────
def dtype_table(torch, dev_type:str):
    # lst=[("FP64",torch.float64),("FP32",torch.float32),
    #      ("FP16",torch.float16),("INT8",torch.int8)]
    lst = [("FP32",torch.float32),("FP16",torch.float16)]
    if hasattr(torch,"float8_e4m3fn"): lst.insert(3,("FP8",torch.float8_e4m3fn))
    if dev_type == "mps":
        # Metal 目前仅支持 FP32、FP16、INT8；移除 FP64 与 FP8
        lst = [p for p in lst if p[0] in ("FP32", "FP16", "INT8")]
    return lst

def rand_tensor(shape,dtype,dev,torch):
    if dtype.is_floating_point and dtype!=torch.int8:
        try: return torch.randn(shape,dtype=dtype,device=dev)
        except Exception: pass
    if dtype==torch.int8:
        try: return torch.randint(-128,128,shape,dtype=torch.int8,device=dev)
        except Exception: pass
    base = torch.randn(shape, dtype=torch.float32, device=dev)
    if dtype == torch.int8:
        return (base * 127).clamp(-128, 127).to(torch.int8)
    try:
        return base.to(dtype)
    except (RuntimeError, TypeError):
        # 后端不支持该 dtype 时退回 fp32 避免崩溃
        return base
    return (base*127).clamp(-128,127).to(torch.int8) if dtype==torch.int8 else base.to(dtype)

def safe_mm(a, b, torch):
    """Try matmul, return True if supported; ensure sync for accurate timing."""
    try:
        _ = a @ b
        if a.device.type == "cuda":
            torch.cuda.synchronize(a.device)
        elif a.device.type == "mps":
            torch.mps.synchronize()
        return True
    except (RuntimeError, TypeError):
        return False

# ─── SINGLE GPU GEMM ─────────────────────────────────────────────────────────
def bench_gpu(N:int,base_iters:int,dev):
    import torch
    name=(torch.cuda.get_device_name(dev) if dev.type=="cuda"
          else get_mps_name() if dev.type=="mps" else str(dev))
    print(f"\n→ GEMM on {dev}: {N}×{N}, base iters={base_iters}\n   Device: {name}")
    scale={"FP16":32,"FP8":1,"INT8":1,"FP32":4,"FP64":1}

    for tag,dt in dtype_table(torch,dev.type):
        it=max(1,int(base_iters*scale[tag]))
        a,b=rand_tensor((N,N),dt,dev,torch),rand_tensor((N,N),dt,dev,torch)
        if not safe_mm(a,b,torch):
            print(f"   {tag}: unsupported"); continue
        t0=time.perf_counter()
        for _ in trange(it): 
            __=a@b
        print("[debug:] finish working!")
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()
        dt=time.perf_counter()-t0
        print(f"   {tag}: {it:5d} it  {dt:.3f}s → {fmt(2*N**3*it/dt)}")
        
    print("Finish Bench GPU.")

# ─── TENSOR CORE GEMM ────────────────────────────────────────────────────────
def bench_tensor_core(N:int,iters:int):
    import torch
    if not torch.cuda.is_available(): return
    maj,_=torch.cuda.get_device_capability()
    if maj<7: return
    dev=torch.device("cuda")
    print(f"\n→ Tensor Core GEMM on {torch.cuda.get_device_name(dev)}")
    torch.backends.cuda.matmul.allow_tf32, old=tf32 = True, torch.backends.cuda.matmul.allow_tf32
    for tag,dt,rep in [("FP16 (TC)",torch.float16,iters),
                       ("TF32 (TC)",torch.float32,max(1,iters//8))]:
        a,b=rand_tensor((N,N),dt,dev,torch),rand_tensor((N,N),dt,dev,torch)
        if not safe_mm(a,b,torch): continue
        t0=time.perf_counter()
        for _ in trange(rep): 
            __=a@b
        torch.cuda.synchronize()
        dt=time.perf_counter()-t0
        print(f"   {tag}: {rep:5d} it  {dt:.3f}s → {fmt(2*N**3*rep/dt)}")
    torch.backends.cuda.matmul.allow_tf32=old

# ─── MULTI-GPU GEMM ──────────────────────────────────────────────────────────
def bench_multi_gpu(N:int,iters:int):
    import torch
    k=torch.cuda.device_count()
    if k<2: 
        print("Only one GPU detected, return.")
        return
    print(f"\n→ Multi-GPU GEMM (×{k}), base iters={iters}")
    bar=threading.Barrier(k); res={}
    scale={"FP16":1,"FP8":1,"INT8":1,"FP32":1/8,"FP64":1/64}

    def worker(rank:int):
        dev=torch.device(f"cuda:{rank}")
        for tag,dt in dtype_table(torch,dev.type):
            it=max(1,int(iters*scale[tag]))
            a,b=rand_tensor((N,N),dt,dev,torch),rand_tensor((N,N),dt,dev,torch)
            if not safe_mm(a,b,torch): continue
            bar.wait(); torch.cuda.synchronize(dev)
            t0=time.perf_counter()
            for _ in trange(it): 
                __=a@b
            torch.cuda.synchronize(dev)
            dt=time.perf_counter()-t0
            res.setdefault(tag,[]).append((it,dt))

    th=[threading.Thread(target=worker,args=(i,)) for i in range(k)]
    [t.start() for t in th]; [t.join() for t in th]
    for tag,vals in res.items():
        it_avg=sum(i for i,_ in vals)//len(vals); dt=sum(d for _,d in vals)/len(vals)
        print(f"   {tag}: {it_avg:5d} it  {dt:.3f}s → {fmt(2*N**3*it_avg*k/dt)}")
    
    print("Finish Bench Multiple GPU.")

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cpu-iters",type=int,default=50_000_000)
    ap.add_argument("--matrix-size",type=int,default=4096)
    ap.add_argument("--matrix-iters",type=int,default=1000)
    args=ap.parse_args()

    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python  : {platform.python_implementation()} {platform.python_version()}")
    ncore=multiprocessing.cpu_count(); print(f"Logical cores: {ncore}\n")

    t1=bench_cpu_single(args.cpu_iters)
    print(f"→ CPU(single)        {t1:.3f}s → {fmt(2*args.cpu_iters/t1)}")
    total=args.cpu_iters*ncore
    t2=bench_cpu_multi(total,ncore)
    print(f"→ CPU({ncore}-core)  {t2:.3f}s → {fmt(2*total/t2)}  Speedup ×{t1/t2*ncore:.2f}")

    try:
        import torch
        if torch.cuda.is_available():
            bench_gpu(args.matrix_size,args.matrix_iters,torch.device("cuda"))
            bench_tensor_core(args.matrix_size,args.matrix_iters)
            bench_multi_gpu(args.matrix_size,args.matrix_iters)
        elif getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
            bench_gpu(args.matrix_size,args.matrix_iters,torch.device("mps"))
        else:
            print("\nNo compatible GPU backend detected.")
    except ImportError:
        print("\nPyTorch not installed; GPU tests skipped.")

if __name__=="__main__":
    main()
