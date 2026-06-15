import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.testing import do_bench
from triton.testing import Benchmark, perf_report
bench_perf_report = None

def autotune_configs():
    warps = 32
    max_threads = 1024
    configs = []

    for num_warps in [1, 2, 4, 8, 16, 32]:
        if num_warps * warps <= max_threads:
            configs.append(triton.Config({}, num_warps=num_warps))
    return configs


@triton.autotune(
        configs=autotune_configs(),
        key=["N"],
)
@triton.jit
def layer_norm_fwd_v1_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, Rstd_ptr,
    stride_x_row, stride_y_row,     # 传入行步长即可灵活索引，Triton编程常见模式
    N, eps, BLOCK_N: tl.constexpr
):
    # 每个program处理1行
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row


    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    
    x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask).to(tl.float32)


    mean = tl.sum(x, axis=0) / N
    tl.store(Mean_ptr + row_idx, mean)

    x_bar = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_bar * x_bar, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd_ptr + row_idx, rstd)

    y = (x - mean) * rstd * w + b
    tl.store(Y_row_ptr + cols, y, mask=mask)


def layer_norm_fwd_v1(x, weight, bias=None, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    if bias is None:
        bias = torch.zeros(N, device=x.device, dtype=weight.dtype)
    if weight is None:
        weight = torch.ones(N, device=x.device, dtype=x.dtype)

    BLOCK_N = triton.next_power_of_2(N)
    layer_norm_fwd_v1_kernel[(M,)](
        x, y, weight, bias, mean, rstd,
        x.stride(0), y.stride(0),
        N, eps, BLOCK_N=BLOCK_N
    )
    return y, mean, rstd

def test_correctness(shapes=[(128, 256), (512, 1024)]):
    for M, N in shapes:
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-5

        ln = nn.LayerNorm(N, eps=eps).to('cuda')
        ln.weight.data = weight
        ln.bias.data = bias
        y_ref = ln(x)

        y_tri, _, _ = layer_norm_fwd_v1(x, weight, bias, eps)

        max_diff = (y_tri - y_ref).abs().max().item()
        print(f"Shape ({M}, {N}): max diff = {max_diff:.6e}")



@perf_report(
    Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["triton", "pytorch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="LayerNorm Fwd Performance",
        args={"M": 1024, "eps": 1e-5, "dtype": torch.float32},
    )
)
def _bench_perf_report(M, N, eps, dtype, provider):
    device = 'cuda'
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)

    if provider == "triton":
        def run():
            return layer_norm_fwd_v1(x, weight, bias, eps)
    else:
        ln = nn.LayerNorm(N, eps=eps).to(device)
        ln.weight.data = weight
        ln.bias.data = bias
        def run():
            return ln(x)
    return do_bench(run, quantiles=[0.5, 0.2, 0.8])

bench_perf_report = _bench_perf_report

if __name__ == "__main__":
    test_correctness()
    if bench_perf_report is not None:
        bench_perf_report.run(show_plots=True, print_data=True, save_path="./layer_norm_fwd")