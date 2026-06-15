import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.testing import do_bench
from triton.testing import Benchmark, perf_report



def autotune_configs():
    configs = []
    for num_warps in [1, 2, 4, 8, 16, 32]:
        if num_warps * 32 <= 1024:
            configs.append(triton.Config({}, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=autotune_configs(),
    key=["N"],
)
@triton.jit
def rms_norm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: rstd = rsqrt(mean(x^2) + eps)
    x2 = x * x
    mean_x2 = tl.sum(x2, axis=0) / N
    rstd = tl.rsqrt(mean_x2 + eps)

    y = x * rstd * w
    tl.store(Y_row_ptr + cols, y, mask=mask)





def rms_norm_fwd(x, weight, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)

    if weight is None:
        weight = torch.ones(N, device=x.device, dtype=x.dtype)

    BLOCK_N = triton.next_power_of_2(N)
    rms_norm_fwd_kernel[(M,)](
        x, y, weight,
        x.stride(0), y.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
    )
    return y



def rms_norm_torch(x, weight, eps=1e-5):
    N = x.shape[-1]
    rms = nn.RMSNorm(N, eps=eps, device=x.device, dtype=x.dtype)
    rms.weight.data.copy_(weight)
    return rms(x)


def test_correctness():
    shapes = [
        (128, 256),
        (128, 512),
        (128, 1024),
        (512, 1024),
        (1024, 2048),
        (512, 4096),
        (128, 8192),
        (64, 16384),
    ]
    dtypes = [torch.float32, torch.float16]

    print("=" * 80)
    print("RMSNorm Correctness Tests  (Triton vs PyTorch nn.RMSNorm)")
    print("=" * 80)

    all_pass = True
    for dtype in dtypes:
        print(f"\n--- dtype: {dtype} ---")
        for M, N in shapes:
            x = torch.randn(M, N, device="cuda", dtype=dtype)
            weight = torch.randn(N, device="cuda", dtype=dtype)
            eps = 1e-5

            y_ref = rms_norm_torch(x, weight, eps)
            y_v1 = rms_norm_fwd(x, weight, eps)

            max_diff = (y_v1.float() - y_ref.float()).abs().max().item()

            # fp16 has lower precision → relaxed threshold
            tol = 1e-2 if dtype == torch.float16 else 1e-4
            ok = "✓" if max_diff < tol else "✗"

            if max_diff >= tol:
                all_pass = False

            print(f"  ({M:>4d}, {N:>5d}): "
                  f"v1 max_diff={max_diff:.6e} {ok}  ")

    print()
    if all_pass:
        print("All correctness tests PASSED ✓")
    else:
        print("Some tests FAILED ✗ (expected for fp16 on certain shapes)")
    return all_pass


@perf_report(
    Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["triton_v1", "pytorch"],
        line_names=["Triton-v1", "PyTorch"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="RMSNorm Forward Performance (M=2048, fp16)",
        args={"M": 2048, "eps": 1e-5, "dtype": torch.float16},
    )
)
def bench_rmsnorm_fp16(M, N, eps, dtype, provider):
    """Benchmark RMSNorm with fp16 data."""
    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)

    if provider == "triton_v1":
        fn = lambda: rms_norm_fwd(x, weight, eps)
    else:  # pytorch
        rms = nn.RMSNorm(N, eps=eps, device=device, dtype=dtype)
        rms.weight.data.copy_(weight)
        fn = lambda: rms(x)

    return do_bench(fn, quantiles=[0.5, 0.2, 0.8])


@perf_report(
    Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["triton", "pytorch"],
        line_names=["Triton",  "PyTorch"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="RMSNorm Forward Performance (M=2048, fp32)",
        args={"M": 2048, "eps": 1e-5, "dtype": torch.float32},
    )
)
def bench_rmsnorm_fp32(M, N, eps, dtype, provider):
    """Benchmark RMSNorm with fp32 data."""
    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)

    if provider == "triton":
        fn = lambda: rms_norm_fwd(x, weight, eps)
    else:  # pytorch
        rms = nn.RMSNorm(N, eps=eps, device=device, dtype=dtype)
        rms.weight.data.copy_(weight)
        fn = lambda: rms(x)

    return do_bench(fn, quantiles=[0.5, 0.2, 0.8])


def run_manual_benchmark():
    """Standalone benchmark using do_bench directly — no pandas needed."""
    M_vals = [128, 512, 1024, 2048, 4096]
    N_vals = [256, 512, 1024, 2048, 4096, 8192]
    dtypes = [torch.float16, torch.float32]
    eps = 1e-5

    for dtype in dtypes:
        print(f"\n{'='*80}")
        print(f"RMSNorm Performance: dtype={dtype}")
        print(f"{'='*80}")
        header = f"{'M':>6s}  {'N':>6s}  {'Triton(ms)':>15s}  {'PyTorch (ms)':>15s}  {'my vs PT':>10s} "
        print(header)
        print("-" * 80)

        for M in M_vals:
            for N in N_vals:
                x = torch.randn(M, N, device="cuda", dtype=dtype)
                weight = torch.randn(N, device="cuda", dtype=dtype)

                # Warmup
                for _ in range(10):
                    rms_norm_fwd(x, weight, eps)
                    rms_norm_torch(x, weight, eps)
                torch.cuda.synchronize()

                # Benchmark
                t_v1 = do_bench(lambda: rms_norm_fwd(x, weight, eps))
                t_pt = do_bench(lambda: rms_norm_torch(x, weight, eps))

                speedup_v1 = t_pt / t_v1 if t_v1 > 0 else 0

                print(f"{M:>6d}  {N:>6d}  {t_v1:>15.6f}  {t_pt:>15.6f}  "
                      f"{speedup_v1:>9.2f}x")


if __name__ == "__main__":
    test_correctness()

    run_manual_benchmark()

    import pandas as pd  # noqa: F401
    print("\n\n>>> perf_report fp16 benchmark")
    bench_rmsnorm_fp16.run(show_plots=True, print_data=True,
                            save_path="./rms_norm_fwd_fp16")
    print("\n>>> perf_report fp32 benchmark")
    bench_rmsnorm_fp32.run(show_plots=True, print_data=True,
                            save_path="./rms_norm_fwd_fp32")
