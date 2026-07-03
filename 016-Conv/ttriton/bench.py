"""Triton Conv2D Benchmark — 正确性 + 性能 + 显存对比"""

import torch, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttriton.common import (
    parse_conv_params, compute_output_dims, gflops, nrmse,
    benchmark_ms, measure_peak_memory,
    build_result_table, build_speedup_table,
)
from ttriton.conv_v0_naive import conv_v0
from ttriton.conv_v1_spatial import conv_v1
from ttriton.conv_v2_tiled import conv_v2
from ttriton.conv_v3_im2col import conv_v3
from ttriton.conv_v4_implicit_gemm import conv_v4
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich import box

console = Console(highlight=False, force_terminal=True)

PROBLEMS = [
    ("ResNet-mid   3×3  C=64→64    56×56 ×16", 16,  64,  56,  56,  64, 3, 3, 0),
    # ("Deep-layer   3×3  C=128→128  28×28 ×32", 32, 128,  28,  28, 128, 3, 3, 0),
    # ("Wide-conv    3×3  C=256→256  28×28 ×8",   8, 256,  28,  28, 256, 3, 3, 0),
    # ("First-layer  7×7  C=3→64    224×224 ×8",  8,   3, 224, 224,  64, 7, 7, 3),
]

# (name, fn) — fn(x, w, b, stride, padding) → NCHW 输出
KERNELS = [
    ("v0-naive",         conv_v0),
    ("v1-空间分块",        conv_v1),
    ("v2-空间+通道分块",    conv_v2),
    ("v3-im2col+GEMM",    conv_v3),
    ("v4-ImplicitGEMM",   conv_v4),
]


def bench_one(name, fn, torch_ref, gflop_val):
    """对单个 kernel 计时+测显存+测误差。fn 为无参 callable"""
    ms = benchmark_ms(fn, ())
    mem, out = measure_peak_memory(fn, ())
    err = nrmse(out, torch_ref)
    gf = gflop_val / (ms / 1000) if ms > 0 else 0
    status = "✓" if err < 1e-3 else f"⚠{err:.1e}"
    return (name, ms, gf, mem, status)


def run():
    device = 'cuda'
    props = torch.cuda.get_device_properties(0)

    console.print()
    console.print(Panel(
        f"[bold bright_white]Triton Conv2D Benchmark[/]\n"
        f"[dim]{props.name}[/]  ·  [dim]sm_{props.major}{props.minor}[/]\n",
        box=box.DOUBLE, border_style="bright_cyan", padding=(1, 3), expand=False))

    all_speedups = []

    for prob_name, N, C, H, W, K, R, S, pad in PROBLEMS:
        sh, sw, ph, pw = parse_conv_params(1, pad)
        H_out, W_out = compute_output_dims(H, W, R, S, sh, sw, ph, pw)
        gflop_val = gflops(N, K, H_out, W_out, C, R, S)
        shape_info = f"{N}×{K}×{H_out}×{W_out}"

        console.print()
        console.print(Rule(title=f"[bold bright_white]{prob_name}[/]", style="bright_black"))

        torch.manual_seed(42)
        x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
        w = torch.randn(K, C, R, S, device=device, dtype=torch.float32)
        b = torch.randn(K, device=device, dtype=torch.float32)

        # PyTorch reference
        torch_fn = lambda: F.conv2d(x, w, b, stride=1, padding=pad)
        torch_ref = torch_fn()
        torch_ms = benchmark_ms(torch_fn, ())
        torch_mem, _ = measure_peak_memory(torch_fn, ())
        torch_gf = gflop_val / (torch_ms / 1000)
        torch_row = ("torch(cuDNN)", torch_ms, torch_gf, torch_mem, "—")

        rows = [bench_one(name, lambda f=fn: f(x, w, b, stride=1, padding=pad),
                          torch_ref, gflop_val)
                for name, fn in KERNELS]
        rows.append(torch_row)

        console.print(build_result_table(prob_name, shape_info, gflop_val, rows))

        # 收集加速比
        baseline = rows[0][1]
        if baseline > 0:
            all_speedups.append(
                [prob_name] + [baseline / r[1] if r[1] > 0 else 0 for r in rows])

    # 加速比总览
    if all_speedups:
        console.print()
        console.print(Rule(title="[bold bright_cyan]Speedup vs v0-naive[/]",
                           style="bright_cyan"))
        col_names = ["Problem"] + [k[0] for k in KERNELS] + ["torch(cuDNN)"]
        console.print(build_speedup_table(all_speedups, col_names))
        console.print()


if __name__ == "__main__":
    run()
