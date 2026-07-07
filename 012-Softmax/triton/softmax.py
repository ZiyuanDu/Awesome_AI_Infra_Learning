import triton
import triton.language as tl
import torch


@triton.jit
def kernel_softmax_fuse(
    x_ptr,
    x_row_stride,
    y_ptr,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
    x = tl.exp(x - tl.max(x))
    eps = float(1e-9)
    x /= tl.maximum(tl.sum(x), eps)
    tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim1_fuse(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_fuse[[n_rows]](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=32,
    )
    return y


@triton.jit
def kernel_softmax_tile(
    x_ptr, x_row_stride,
    y_ptr, y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    CACHE_OPT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride

    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")

    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        mm = tl.maximum(mm, x)
    mm = tl.max(mm)

    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if CACHE_OPT:
        for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm)
            ss += x
    else:
        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm)
            ss += x

    ss = tl.sum(ss)
    eps = float(1e-9)
    ss = tl.maximum(ss, eps)

    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        x = tl.exp(x - mm) / ss
        tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim1_tile(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_tile[[n_rows]](
        x, x.stride(0),
        y, y.stride(0),
        n_cols,
        BLOCK_SIZE=2**14,
        CACHE_OPT=cache_opt,
        num_warps=32,
    )
    return y


@triton.jit
def kernel_softmax_online(
    x_ptr,
    x_row_stride,
    y_ptr,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    CACHE_OPT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride

    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        mm_new = tl.maximum(mm, x)
        if i:  # 第 1 轮不需要，且容易整出 nan
            ss *= tl.exp(mm - mm_new)
        x = tl.exp(x - mm_new)
        ss += tl.where(idx < n_cols, x, 0.0)
        mm = mm_new

    mm_new = tl.max(mm)
    ss *= tl.exp(mm - mm_new)
    ss = tl.sum(ss)
    mm = mm_new

    eps = float(1e-9)
    ss = tl.maximum(ss, eps)

    if CACHE_OPT:
        for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask=idx < n_cols)
    else:
        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim1_online(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_online[[n_rows]](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=2**12,
        CACHE_OPT=cache_opt,
        num_warps=32,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=32, num_stages=3),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32, num_stages=4),
    ],
    key=["n_cols"],
)
@triton.jit
def kernel_softmax_online_v3(
    x_ptr, x_row_stride, y_ptr, y_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr, 
    REQUIRES_MASK: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride

    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    n_tiles = tl.cdiv(n_cols, BLOCK_SIZE)
    
    # --- Pass 1: Max & Sum ---
    for i in range(0, n_tiles):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        if REQUIRES_MASK:
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf")).to(tl.float32)
        else:
            x = tl.load(x_ptr + idx).to(tl.float32)

        mm_new = tl.maximum(mm, x)
        if i > 0:
            ss *= tl.exp(mm - mm_new)
        x = tl.exp(x - mm_new)
        
        if REQUIRES_MASK:
            ss += tl.where(idx < n_cols, x, 0.0)
        else:
            ss += x
        mm = mm_new

    # 跨线程归约
    mm_new = tl.max(mm)
    ss *= tl.exp(mm - mm_new)
    ss = tl.sum(ss)
    mm = mm_new
    
    # !!! 关键优化：除法变倒数乘法 !!!
    inv_ss = 1.0 / tl.maximum(ss, float(1e-9))

    # --- Pass 2: Write ---
    # 倒序遍历最大化 L2 Cache 命中率
    for i in range(n_tiles - 1, -1, -1):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        if REQUIRES_MASK:
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf")).to(tl.float32)
            out = (tl.exp(x - mm) * inv_ss)  # 用乘法替代除法
            tl.store(y_ptr + idx, out, mask=idx < n_cols)
        else:
            x = tl.load(x_ptr + idx).to(tl.float32)
            out = (tl.exp(x - mm) * inv_ss)
            tl.store(y_ptr + idx, out)


def triton_softmax_dim1_online_v3(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    
    # 由于 autotune 提供的 BLOCK_SIZE 都在 1024 ~ 8192 之间，
    # 若 n_cols 是 8192 的倍数，则必然是不需要 Mask 的，借此省去循环内的判断开销
    requires_mask = (n_cols % 8192 != 0)

    kernel_softmax_online_v3[(n_rows,)](
        x, x.stride(0), y, y.stride(0),
        n_cols, REQUIRES_MASK=requires_mask,
    )
    return y


def test():
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand([2**10, 2**15], device=DEVICE)
    mp = {
        "torch": lambda: torch.softmax(x, dim=1),
        "triton_fuse": lambda: triton_softmax_dim1_fuse(x),
        "triton_tile_no_cache": lambda: triton_softmax_dim1_tile(x, cache_opt=False),
        "triton_tile": lambda: triton_softmax_dim1_tile(x),
        "triton_online_no_cache": lambda: triton_softmax_dim1_online(
            x, cache_opt=False
        ),
        "triton_online": lambda: triton_softmax_dim1_online(x),
        "triton_online_v3": lambda: triton_softmax_dim1_online_v3(x),
    }
    y_torch = mp["torch"]()
    for k, v in mp.items():
        y_triton = v()
        print("{}: Maxdiff is {}".format(k, torch.max(torch.abs(y_torch - y_triton))))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_col"],
        x_vals=[2**i for i in range(8, 19)],  # triton maximum tensor numel (131072)
        line_arg="provider",
        line_vals=[
            "torch",
            "triton_fuse",
            "triton_tile",
            "triton_online",
            "triton_online_v3",
        ],
        line_names=[
            "Torch",
            "Triton_fuse",
            "Triton_tile",
            "Triton_online",
            "Triton_v3 ",
        ],
        plot_name="softmax-time",
        args={},
    )
)
def benchmark(n_col, provider):
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand([2**11, n_col], device=DEVICE)
    mp = {
        "torch": lambda: torch.softmax(x, dim=1),
        "triton_fuse": lambda: triton_softmax_dim1_fuse(x),
        "triton_tile": lambda: triton_softmax_dim1_tile(x),
        "triton_online": lambda: triton_softmax_dim1_online(x),
        "triton_online_v3": lambda: triton_softmax_dim1_online_v3(x),
    }
    return triton.testing.do_bench(mp[provider])  # ms


if __name__ == "__main__":
    torch.manual_seed(3407)
    test()
    benchmark.run(print_data=True, show_plots=False, save_path=".")