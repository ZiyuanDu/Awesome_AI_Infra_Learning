import torch
import torch.nn.functional as F
from rich.table import Table
from rich.text import Text
from rich import box


def parse_conv_params(stride, padding):
    """解析 stride/padding → (sh, sw), (ph, pw)"""
    sh = stride if isinstance(stride, int) else stride[0]
    sw = stride if isinstance(stride, int) else stride[1]
    ph = padding if isinstance(padding, int) else padding[0]
    pw = padding if isinstance(padding, int) else padding[1]
    return sh, sw, ph, pw


def compute_output_dims(H_in, W_in, kH, kW, sh, sw, ph, pw):
    """计算输出空间尺寸"""
    return (H_in + 2 * ph - kH) // sh + 1, (W_in + 2 * pw - kW) // sw + 1


def pad_input(x, ph, pw):
    """zero-padding"""
    if ph > 0 or pw > 0:
        return F.pad(x, (pw, pw, ph, ph), mode='constant', value=0.0)
    return x


def nchw_from_gemm(out_2d, B, C_out, H_out, W_out):
    """GEMM 输出 [C_out, B×H_out×W_out] → NCHW [B, C_out, H_out, W_out]"""
    return out_2d.reshape(C_out, B, H_out, W_out).permute(1, 0, 2, 3).contiguous()


def gflops(N, C_out, H_out, W_out, C_in, kH, kW):
    """理论计算量 GFLOPS = 2 × N×K×OH×OW×C×R×S × 1e-9"""
    return 2.0 * N * C_out * H_out * W_out * C_in * kH * kW * 1e-9


def nrmse(a, b):
    """归一化均方根误差"""
    mse = (a - b).square().mean()
    ref = b.square().mean()
    return (mse / ref).sqrt().item() if ref > 0 else float('inf')


def assert_close(out, ref, tol=1e-3, name=""):
    """ 统一的正确性校验 """
    err = nrmse(out, ref)
    assert err < tol, f"{name} 正确性检查失败! nrmse={err:.2e} ≥ {tol:.0e}"
    return err


def benchmark_ms(fn, args, warmup=10, iters=50):
    """CUDA event 计时，返回平均毫秒。prerun 触发 autotune 缓存。"""
    fn(*args); torch.cuda.synchronize()        # 触发 autotuning
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def measure_peak_memory(fn, args):
    """测量 fn 执行期间的额外峰值显存 (MiB)"""
    fn(*args); torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = fn(*args)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024 ** 2), result



def _fmt_val(val):
    """数值格式化: 自动选 .4f / .1f / .2e"""
    if isinstance(val, float):
        if val == 0: return "—"
        if abs(val) < 0.01: return f"{val:.2e}"
        return f"{val:.4f}" if val < 100 else f"{val:.1f}"
    return str(val)


def _status_icon(status):
    """状态图标着色"""
    if status == "✓":  return Text("✓", style="bold green")
    if status == "—":  return Text("—", style="dim")
    if status.startswith("⚠"): return Text(status, style="bold yellow")
    if status.startswith("✗"): return Text(status, style="bold red")
    return Text(status)


def _best_idx(ms_list):
    """返回最快 kernel 的索引（排除 torch 行）"""
    valid = [(i, m) for i, m in enumerate(ms_list) if m > 0]
    return min(valid, key=lambda t: t[1])[0] if valid else None


def build_result_table(title, shape_info, gflop_val, rows):
    """构建单个 problem 的 Rich 结果表格"""
    table = Table(
        box=box.SIMPLE_HEAD, show_lines=False, pad_edge=False, expand=False,
        title=f"[bold bright_white]{title}[/]",
        caption=f"[dim]{shape_info}  ·  {gflop_val:.3f} GFLOPS[/]",
        caption_justify="left",
        border_style="bright_black",
        header_style="bold bright_cyan",
    )
    table.add_column("Kernel", style="bright_white", min_width=22, no_wrap=True)
    table.add_column("Time (ms)", justify="right", min_width=10)
    table.add_column("GFLOPS", justify="right", min_width=10)
    table.add_column("Peak Mem", justify="right", min_width=10)
    table.add_column("Status", justify="center", min_width=8)

    ms_vals = [r[1] for r in rows]
    best = _best_idx(ms_vals)
    torch_i = len(rows) - 1

    for i, (name, ms, gf, mem, status) in enumerate(rows):
        is_torch = (i == torch_i)
        is_best = (i == best)

        kw = {"style": "bold bright_green"} if is_best else \
             ({"style": "bold bright_yellow"} if is_torch else {})
        time_kw = {"style": "bold bright_green"} if is_best else \
                  ({"style": "bright_yellow"} if is_torch else {})

        table.add_row(
            Text(name, **kw),
            Text(f"{ms:.4f}" if ms > 0 else "—", **time_kw),
            Text(f"{gf:.1f}" if gf > 0 else "—", **kw),
            Text(f"{mem:.1f}" if mem > 0 else "—"),
            _status_icon(status),
        )
    return table


def build_speedup_table(rows, col_names):
    """构建加速比矩阵 Rich 表格"""
    table = Table(
        box=box.HEAVY_HEAD, show_lines=False, expand=False,
        title="[bold bright_cyan]Speedup vs v0-naive[/]",
        title_justify="left",
        border_style="bright_black",
        header_style="bold bright_cyan",
    )
    table.add_column("Problem", style="bright_white", min_width=30, no_wrap=True)
    for c in col_names[1:]:
        table.add_column(c, justify="center", min_width=8)

    for row in rows:
        cells = [Text(str(row[0]), style="bright_white")]
        for v in row[1:]:
            if v <= 0:          cells.append(Text("—", style="dim"))
            elif v >= 1.5:      cells.append(Text(f"{v:.1f}x", style="bold bright_green"))
            elif v >= 1.0:      cells.append(Text(f"{v:.1f}x", style="bright_white"))
            else:               cells.append(Text(f"{v:.1f}x", style="yellow"))
        table.add_row(*cells)
    return table
