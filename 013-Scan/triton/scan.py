"""
scan.py — Triton Parallel Scan (Prefix Sum)

三趟法实现:
  Pass 1 (reduce_kernel):   每个 block 计算其内部总和 → sums[]
  Pass 2 (scan_sums_kernel): 对 sums[] 做 associative_scan → 得每个 block 的前缀偏移
  Pass 3 (scan_add_kernel):  对每个 block 做局部 scan + 加上全局偏移 → 输出

参考: 013-Scan CUDA 版本的多 block 递归分解策略
"""

import torch
import triton
import triton.language as tl

@triton.jit
def combine(a, b):
    return a + b

# Pass 1: 仅计算每个 Block 的总和
@triton.jit
def reduce_kernel(x_ptr, sums_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)
    tl.store(sums_ptr + pid, tl.reduce(x, 0, combine))

# Pass 2: 计算 Block 级别的全局偏移量
@triton.jit
def scan_sums_kernel(sums_ptr, offsets_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    sums = tl.load(sums_ptr + offsets, mask=mask, other=0.0)
    scanned = tl.associative_scan(sums, axis=0, combine_fn=combine)
    tl.store(offsets_ptr + offsets, scanned - sums, mask=mask)

# Pass 3: 在寄存器中局部 Scan 并加上偏移量
@triton.jit
def scan_add_kernel(x_ptr, y_ptr, offsets_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scanned = tl.associative_scan(x, axis=0, combine_fn=combine)
    offset = tl.load(offsets_ptr + pid)  # 读取此 block 的全局前缀偏移
    tl.store(y_ptr + offsets, scanned + offset, mask=mask)

def scan(x: torch.Tensor) -> torch.Tensor:
    """Exclusive scan (prefix sum) using Triton three-pass algorithm."""
    n = x.numel()
    y = torch.empty_like(x)
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    sums = torch.empty(num_blocks, device=x.device, dtype=x.dtype)
    offsets = torch.empty(num_blocks, device=x.device, dtype=x.dtype)

    grid = (num_blocks,)
    reduce_kernel[grid](x, sums, n, BLOCK_SIZE)

    scan_bs = max(16, triton.next_power_of_2(num_blocks))
    scan_sums_kernel[(1,)](sums, offsets, num_blocks, scan_bs)

    scan_add_kernel[grid](x, y, offsets, n, BLOCK_SIZE)
    return y


if __name__ == "__main__":
    # 正确性测试: 与 torch.cumsum 比较
    N = 10000
    x = torch.randn(N, device='cuda')
    y_triton = scan(x)

    # torch.cumsum 是 inclusive scan，转为 exclusive 需要右移并在前面填 0
    y_torch = torch.cumsum(x, dim=0)
    y_torch = torch.cat([torch.zeros(1, device='cuda'), y_torch[:-1]])

    max_error = (y_triton - y_torch).abs().max().item()
    print(f"N = {N}, Max error: {max_error:.6e}")
    print("PASS" if max_error < 1e-4 else "FAIL")

    # 性能测试
    print("\n--- Triton Scan Performance ---")
    for n in [1024, 4096, 16384, 65536, 1 << 20, 16 << 20]:
        x = torch.randn(n, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(5):
            _ = scan(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            _ = scan(x)
        end.record()
        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / 10
        bw = (2 * n * 4) / (ms * 1e6)  # GB/s
        print(f"  N={n:7d}  |  {ms:8.4f} ms  |  {bw:8.2f} GB/s")
