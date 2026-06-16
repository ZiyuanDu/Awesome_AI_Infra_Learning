"""
matmul.py — Triton SGEMM: naive → tiled → autotuned, with benchmarks.

Usage:
    python matmul.py                # correctness + quick benchmark
    python matmul.py --bench-only   # skip correctness, benchmark only
    python matmul.py --plot         # generate perf report plots (needs pandas)

Requirements:
    pip install torch triton
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

# ---------------------------------------------------------------------------
# Triton kernels: naive → tiled → autotuned
# ---------------------------------------------------------------------------

@triton.jit
def matmul_naive_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Naive: one program per output element, no shared memory blocking."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def matmul_tiled_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Tiled: shared memory blocking with L2 cache swizzling via GROUP_M."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def _autotune_configs():
    configs = []
    for BM in [64, 128]:
        for BN in [64, 128]:
            for BK in [32, 64]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 3, 4]:
                        configs.append(triton.Config(
                            {'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK,
                             'GROUP_M': 8},
                            num_warps=num_warps, num_stages=num_stages))
    return configs


@triton.autotune(configs=_autotune_configs(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Autotuned tiled matmul — same logic as matmul_tiled_kernel."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------

def triton_matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_naive_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return c


def triton_matmul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    BM, BN, BK = 128, 128, 32
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_tiled_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
    )
    return c


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Autotuned Triton matmul (warmup + eager)."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_correctness():
    shapes = [
        (128, 256, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 512),
        (2048, 256, 256),
        (256, 2048, 256),
    ]
    dtypes = [torch.float32, torch.float16]

    print("=" * 72)
    print("Triton SGEMM — Correctness Tests")
    print("=" * 72)

    all_pass = True
    for dtype in dtypes:
        print(f"\n  dtype: {dtype}")
        print(f"  {'Shape':>20s}  {'Naive err':>12s}  {'Tiled err':>12s}  {'Auto err':>12s}")
        print(f"  {'-'*60}")
        for M, N, K in shapes:
            a = torch.randn(M, K, device='cuda', dtype=dtype)
            b = torch.randn(K, N, device='cuda', dtype=dtype)
            ref = torch.mm(a.float(), b.float())

            c_naive = triton_matmul_naive(a, b)
            c_tiled = triton_matmul_tiled(a, b)
            c_auto  = triton_matmul(a, b)

            e_n = (c_naive.float() - ref).abs().max().item()
            e_t = (c_tiled.float() - ref).abs().max().item()
            e_a = (c_auto.float() - ref).abs().max().item()

            tol = 1e-1 if dtype == torch.float16 else 1e-3
            p_n = "✓" if e_n < tol else "✗"
            p_t = "✓" if e_t < tol else "✗"
            p_a = "✓" if e_a < tol else "✗"
            if e_n >= tol or e_t >= tol or e_a >= tol:
                all_pass = False

            print(f"  ({M:>4d},{N:>4d},{K:>4d})  {e_n:>10.6f} {p_n}  "
                  f"{e_t:>10.6f} {p_t}  {e_a:>10.6f} {p_a}")

    print(f"\n  {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks():
    configs = [
        # (M, N, K, label)
        (256, 256, 256, "tiny-square"),
        (512, 512, 512, "small-square"),
        (1024, 1024, 1024, "1k-square"),
        (2048, 2048, 2048, "2k-square"),
        (4096, 4096, 4096, "4k-square"),
        (16384, 256, 256, "tall-M16k"),
        (256, 16384, 256, "wide-N16k"),
    ]

    dtypes = [torch.float16, torch.float32]

    for dtype in dtypes:
        print(f"\n{'='*90}")
        print(f"Triton SGEMM Performance — dtype={dtype}")
        print(f"{'='*90}")
        hdr = (f"  {'Config':>16s}  {'Shape':>18s}  "
               f"{'Naive(ms)':>10s}  {'Tiled(ms)':>10s}  "
               f"{'Auto(ms)':>10s}  {'Torch(ms)':>10s}  {'vs Torch':>9s}")
        print(hdr)
        print("  " + "-" * 86)

        for M, N, K, label in configs:
            a = torch.randn(M, K, device='cuda', dtype=dtype)
            b = torch.randn(K, N, device='cuda', dtype=dtype)

            # Warmup
            if M <= 1024:
                for _ in range(10):
                    triton_matmul_naive(a, b); triton_matmul_tiled(a, b)
                    triton_matmul(a, b); torch.mm(a, b)
            torch.cuda.synchronize()

            reps = 200 if M <= 1024 else 50
            t_n = do_bench(lambda: triton_matmul_naive(a, b), rep=reps)
            t_t = do_bench(lambda: triton_matmul_tiled(a, b), rep=reps)
            t_a = do_bench(lambda: triton_matmul(a, b), rep=reps)
            t_pt = do_bench(lambda: torch.mm(a, b), rep=reps)

            speedup = t_pt / t_a if t_a > 0 else 0
            print(f"  {label:>16s}  ({M:>5d},{N:>5d},{K:>5d})  "
                  f"{t_n*1000:>8.4f}   {t_t*1000:>8.4f}   "
                  f"{t_a*1000:>8.4f}   {t_pt*1000:>8.4f}   "
                  f"{speedup:>7.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Triton SGEMM benchmark")
    parser.add_argument("--bench-only", action="store_true",
                        help="Skip correctness tests")
    parser.add_argument("--plot", action="store_true",
                        help="Generate perf-report plots (needs pandas)")
    args = parser.parse_args()

    if not args.bench_only:
        test_correctness()

    run_benchmarks()

    if args.plot:
        try:
            from triton.testing import Benchmark, perf_report

            @perf_report(
                Benchmark(
                    x_names=["M"],
                    x_vals=[256, 512, 1024, 2048, 4096],
                    line_arg="provider",
                    line_vals=["triton_naive", "triton_tiled", "triton_auto", "pytorch"],
                    line_names=["Triton-Naive", "Triton-Tiled", "Triton-Auto", "PyTorch"],
                    ylabel="TFLOPS",
                    plot_name="SGEMM Performance (square, fp16)",
                    args={"N": None, "K": None, "dtype": torch.float16},
                )
            )
            def bench_square(M, N, K, dtype, provider):
                N = K = M
                a = torch.randn(M, K, device="cuda", dtype=dtype)
                b = torch.randn(K, N, device="cuda", dtype=dtype)
                if provider == "triton_naive":
                    fn = lambda: triton_matmul_naive(a, b)
                elif provider == "triton_tiled":
                    fn = lambda: triton_matmul_tiled(a, b)
                elif provider == "triton_auto":
                    fn = lambda: triton_matmul(a, b)
                else:
                    fn = lambda: torch.mm(a, b)
                ms = do_bench(fn)
                tflops = 2.0 * M * N * K / (ms * 1e9)
                return tflops

            bench_square.run(show_plots=True, print_data=True,
                             save_path="./triton_sgemm_perf")
        except ImportError:
            print("[plot] pandas not available — skipping perf_report")
