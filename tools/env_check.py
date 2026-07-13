#!/usr/bin/env python3
"""Report what this machine can do: devices, precision support, rough speed.

Usage: bin/ember env

Run once per machine (Mac, 2080S desktop, 4070 box, rented GPUs) so device
and dtype choices in configs are informed, not guessed. The matmul number is
order-of-magnitude only — honest GPU timing (warmup, events, cache effects)
is a whole topic, and it arrives in forge B0.
"""

import time

import torch


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench_matmul(device: str, dtype, n: int = 2048, iters: int = 20):
    try:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        for _ in range(3):
            _ = a @ b
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        sync(device)
        dt = time.perf_counter() - t0
    except Exception as e:
        print(f"  matmul {str(dtype).replace('torch.', ''):9}: failed ({e})")
        return
    tflops = 2 * n**3 * iters / dt / 1e12
    print(f"  matmul {str(dtype).replace('torch.', ''):9}: ~{tflops:5.1f} TFLOP/s")


def main():
    print(f"torch {torch.__version__}")

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        bf16 = torch.cuda.is_bf16_supported()
        print(f"\ncuda: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
        print(
            f"  bf16 supported: {bf16}"
            + ("" if bf16 else "   -> use fp16 + GradScaler on this GPU")
        )
        bench_matmul("cuda", torch.float32)
        bench_matmul("cuda", torch.bfloat16 if bf16 else torch.float16)
    else:
        print("\ncuda: not available")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("\nmps: available (develop here — fp32, tiny configs)")
        bench_matmul("mps", torch.float32)
    else:
        print("\nmps: not available")

    print("\ncpu:")
    bench_matmul("cpu", torch.float32)


if __name__ == "__main__":
    main()
