import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel_v0(
    input_ptr, weight_ptr, output_ptr,
    N, C, H, W, K, R, S,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_k, stride_wt_c, stride_wt_r, stride_wt_s,
    stride_out_n, stride_out_k, stride_out_h, stride_out_w,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """
    朴素tiled卷积: 每个program处理output的一个空间tile + K维度分块
    input[N][C][H][W] * weight[K][C][R][S] -> output[N][K][OH][OW]
    """
    OH = H - R + 1
    OW = W - S + 1

    # program_id -> (n, k_block, oh_block, ow_block)
    num_oh_blocks = tl.cdiv(OH, BLOCK_OH)
    num_ow_blocks = tl.cdiv(OW, BLOCK_OW)
    num_k_blocks = tl.cdiv(K, BLOCK_K)

    pid = tl.program_id(0)
    n = pid // (num_k_blocks * num_oh_blocks * num_ow_blocks)
    resid = pid % (num_k_blocks * num_oh_blocks * num_ow_blocks)
    k_block = resid // (num_oh_blocks * num_ow_blocks)
    resid = resid % (num_oh_blocks * num_ow_blocks)
    oh_block = resid // num_ow_blocks
    ow_block = resid % num_ow_blocks

    # current block's K range
    k_start = k_block * BLOCK_K
    k_offs = k_start + tl.arange(0, BLOCK_K)

    # current block's spatial range
    oh_offs = oh_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow_offs = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)

    # accumulator: [BLOCK_OH, BLOCK_OW, BLOCK_K]
    acc = tl.zeros((BLOCK_OH, BLOCK_OW, BLOCK_K), dtype=tl.float32)

    # 沿C维度分块累加 (C通常很大, 需要tiling)
    for cb in range(0, C, BLOCK_C):
        c_remain = C - cb
        c_offs = cb + tl.arange(0, BLOCK_C)

        # 加载input[c, oh+r, ow+s] — 对R×S window做展开
        for r in range(R):
            ih_offs = oh_offs + r  # [BLOCK_OH]
            for s in range(S):
                iw_offs = ow_offs + s  # [BLOCK_OW]

                # input[n, c, ih, iw]: shape [BLOCK_C, BLOCK_OH, BLOCK_OW]
                in_ptrs = (
                    input_ptr
                    + n * stride_in_n
                    + c_offs[:, None, None] * stride_in_c
                    + ih_offs[None, :, None] * stride_in_h
                    + iw_offs[None, None, :] * stride_in_w
                )

                in_mask = (
                    (c_offs[:, None, None] < C)
                    & (ih_offs[None, :, None] < H)
                    & (iw_offs[None, None, :] < W)
                )
                inp = tl.load(in_ptrs, mask=in_mask, other=0.0)

                # weight[k, c, r, s]: shape [BLOCK_K, BLOCK_C]
                wt_ptrs = (
                    weight_ptr
                    + k_offs[:, None] * stride_wt_k
                    + c_offs[None, :] * stride_wt_c
                    + r * stride_wt_r
                    + s * stride_wt_s
                )

                wt_mask = (k_offs[:, None] < K) & (c_offs[None, :] < C)
                wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

                # inp: [BLOCK_C, BLOCK_OH, BLOCK_OW]
                # wt:  [BLOCK_K, BLOCK_C]
                # -> acc[oh, ow, k] += sum_c wt[k, c] * inp[c, ih, iw]
                dot_2d = tl.dot(wt, tl.reshape(inp, (BLOCK_C, BLOCK_OH * BLOCK_OW)))
                dot_3d = tl.reshape(dot_2d, (BLOCK_K, BLOCK_OH, BLOCK_OW))
                # [K, OH, OW] -> [OH, K, OW] -> [OH, OW, K]
                acc += tl.trans(tl.trans(dot_3d, 0, 1), 1, 2)

    # 写回output[n, k, oh, ow]
    out_ptrs = (
        output_ptr
        + n * stride_out_n
        + k_offs[None, None, :] * stride_out_k
        + oh_offs[:, None, None] * stride_out_h
        + ow_offs[None, :, None] * stride_out_w
    )

    out_mask = (
        (oh_offs[:, None, None] < OH)
        & (ow_offs[None, :, None] < OW)
        & (k_offs[None, None, :] < K)
    )
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def conv2d_kernel_v1(
    input_ptr, weight_ptr, output_ptr,
    N, C, H, W, K, R, S,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_k, stride_wt_c, stride_wt_r, stride_wt_s,
    stride_out_n, stride_out_k, stride_out_h, stride_out_w,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """
    v1: im2col tiled — 先展开R×S到空间维度 (im2col), 再用类似matmul的方式累加
    与v0的区别: 用tl.dot一次性做C维度的完整归约, 减少循环次数
    """
    OH = H - R + 1
    OW = W - S + 1
    CRS = C * R * S

    num_oh_blocks = tl.cdiv(OH, BLOCK_OH)
    num_ow_blocks = tl.cdiv(OW, BLOCK_OW)
    num_k_blocks = tl.cdiv(K, BLOCK_K)

    pid = tl.program_id(0)
    n = pid // (num_k_blocks * num_oh_blocks * num_ow_blocks)
    resid = pid % (num_k_blocks * num_oh_blocks * num_ow_blocks)
    k_block = resid // (num_oh_blocks * num_ow_blocks)
    resid = resid % (num_oh_blocks * num_ow_blocks)
    oh_block = resid // num_ow_blocks
    ow_block = resid % num_ow_blocks

    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    oh_offs = oh_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow_offs = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)

    # 对im2col的CRS维度进行tiling
    acc = tl.zeros((BLOCK_OH, BLOCK_OW, BLOCK_K), dtype=tl.float32)

    for crs_block in range(0, CRS, BLOCK_C):
        crs_offs = crs_block + tl.arange(0, BLOCK_C)
        crs_mask = crs_offs < CRS

        # 解析crs -> (c, r, s)
        c_offs = crs_offs // (R * S)
        rs_offs = crs_offs % (R * S)
        r_offs = rs_offs // S
        s_offs = rs_offs % S

        # im2col: 输入按(c, r, s)索引 [BLOCK_C, BLOCK_OH, BLOCK_OW]
        ih_offs = oh_offs[None, :, None] + r_offs[:, None, None]  # [BLOCK_C, BLOCK_OH, 1]
        iw_offs = ow_offs[None, None, :] + s_offs[:, None, None]  # [BLOCK_C, 1, BLOCK_OW]

        in_ptrs = (
            input_ptr
            + n * stride_in_n
            + c_offs[:, None, None] * stride_in_c
            + ih_offs * stride_in_h
            + iw_offs * stride_in_w
        )

        in_mask = (
            crs_mask[:, None, None]
            & (ih_offs < H)
            & (iw_offs < W)
        )
        col = tl.load(in_ptrs, mask=in_mask, other=0.0)

        # weight[k, crs]: [BLOCK_K, BLOCK_C]
        wt_ptrs = (
            weight_ptr
            + k_offs[:, None] * stride_wt_k
            + crs_offs[None, :]
        )
        wt_mask = (k_offs[:, None] < K) & crs_mask[None, :]
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        # col: [BLOCK_C, BLOCK_OH, BLOCK_OW] -> [BLOCK_C, BLOCK_OH*BLOCK_OW]
        # wt:  [BLOCK_K, BLOCK_C]
        # acc += wt @ col -> [BLOCK_K, BLOCK_OH*BLOCK_OW]
        dot_2d = tl.dot(wt, tl.reshape(col, (BLOCK_C, BLOCK_OH * BLOCK_OW)))
        dot_3d = tl.reshape(dot_2d, (BLOCK_K, BLOCK_OH, BLOCK_OW))
        # [K, OH, OW] -> [OH, OW, K]
        acc += tl.trans(tl.trans(dot_3d, 0, 1), 1, 2)

    # 写回
    out_ptrs = (
        output_ptr
        + n * stride_out_n
        + k_offs[None, None, :] * stride_out_k
        + oh_offs[:, None, None] * stride_out_h
        + ow_offs[None, :, None] * stride_out_w
    )
    out_mask = (
        (oh_offs[:, None, None] < OH)
        & (ow_offs[None, :, None] < OW)
        & (k_offs[None, None, :] < K)
    )
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    version: int = 0,
    BLOCK_OH: int = 16,
    BLOCK_OW: int = 16,
    BLOCK_K: int = 32,
    BLOCK_C: int = 64,
) -> torch.Tensor:
    """
    Triton 2D convolution (stride=1, pad=0).

    Args:
        input:  [N, C, H, W]
        weight: [K, C, R, S]
        version: 0=naive tiled, 1=im2col-style
    """
    assert input.ndim == 4 and weight.ndim == 4
    assert input.shape[1] == weight.shape[1], \
        f"Channel mismatch: {input.shape[1]} vs {weight.shape[1]}"

    N, C, H, W = input.shape
    K, _, R, S = weight.shape
    OH, OW = H - R + 1, W - S + 1
    assert OH > 0 and OW > 0, f"Output too small: OH={OH}, OW={OW}"

    output = torch.empty((N, K, OH, OW), device=input.device, dtype=input.dtype)

    num_blocks = N * triton.cdiv(K, BLOCK_K) * triton.cdiv(OH, BLOCK_OH) * triton.cdiv(OW, BLOCK_OW)
    grid = (num_blocks,)

    kernel_fn = conv2d_kernel_v0 if version == 0 else conv2d_kernel_v1

    kernel_fn[grid](
        input, weight, output,
        N, C, H, W, K, R, S,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_OH=BLOCK_OH, BLOCK_OW=BLOCK_OW,
        BLOCK_K=BLOCK_K, BLOCK_C=BLOCK_C,
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 8, 'BLOCK_K': 16, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 8, 'BLOCK_K': 32, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_K': 16, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_K': 32, 'BLOCK_C': 64}, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W', 'K', 'R', 'S'],
)
@triton.jit
def conv2d_kernel_autotuned(
    input_ptr, weight_ptr, output_ptr,
    N, C, H, W, K, R, S,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_k, stride_wt_c, stride_wt_r, stride_wt_s,
    stride_out_n, stride_out_k, stride_out_h, stride_out_w,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Autotuned version of conv2d — same algorithm as v1 (im2col-style)."""
    OH = H - R + 1
    OW = W - S + 1
    CRS = C * R * S

    num_oh_blocks = tl.cdiv(OH, BLOCK_OH)
    num_ow_blocks = tl.cdiv(OW, BLOCK_OW)
    num_k_blocks = tl.cdiv(K, BLOCK_K)

    pid = tl.program_id(0)
    n = pid // (num_k_blocks * num_oh_blocks * num_ow_blocks)
    resid = pid % (num_k_blocks * num_oh_blocks * num_ow_blocks)
    k_block = resid // (num_oh_blocks * num_ow_blocks)
    resid = resid % (num_oh_blocks * num_ow_blocks)
    oh_block = resid // num_ow_blocks
    ow_block = resid % num_ow_blocks

    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    oh_offs = oh_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow_offs = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)

    acc = tl.zeros((BLOCK_OH, BLOCK_OW, BLOCK_K), dtype=tl.float32)

    for crs_block in range(0, CRS, BLOCK_C):
        crs_offs = crs_block + tl.arange(0, BLOCK_C)
        crs_mask = crs_offs < CRS

        c_offs = crs_offs // (R * S)
        rs_offs = crs_offs % (R * S)
        r_offs = rs_offs // S
        s_offs = rs_offs % S

        ih_offs = oh_offs[None, :, None] + r_offs[:, None, None]
        iw_offs = ow_offs[None, None, :] + s_offs[:, None, None]

        in_ptrs = (
            input_ptr
            + n * stride_in_n
            + c_offs[:, None, None] * stride_in_c
            + ih_offs * stride_in_h
            + iw_offs * stride_in_w
        )
        in_mask = crs_mask[:, None, None] & (ih_offs < H) & (iw_offs < W)
        col = tl.load(in_ptrs, mask=in_mask, other=0.0)

        wt_ptrs = weight_ptr + k_offs[:, None] * stride_wt_k + crs_offs[None, :]
        wt_mask = (k_offs[:, None] < K) & crs_mask[None, :]
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        dot_2d = tl.dot(wt, tl.reshape(col, (BLOCK_C, BLOCK_OH * BLOCK_OW)))
        dot_3d = tl.reshape(dot_2d, (BLOCK_K, BLOCK_OH, BLOCK_OW))
        # [K, OH, OW] -> [OH, OW, K]
        acc += tl.trans(tl.trans(dot_3d, 0, 1), 1, 2)

    out_ptrs = (
        output_ptr
        + n * stride_out_n
        + k_offs[None, None, :] * stride_out_k
        + oh_offs[:, None, None] * stride_out_h
        + ow_offs[None, :, None] * stride_out_w
    )
    out_mask = (
        (oh_offs[:, None, None] < OH)
        & (ow_offs[None, :, None] < OW)
        & (k_offs[None, None, :] < K)
    )
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def conv2d_autotuned(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Autotuned Triton conv2d."""
    assert input.ndim == 4 and weight.ndim == 4
    N, C, H, W = input.shape
    K, _, R, S = weight.shape
    OH, OW = H - R + 1, W - S + 1

    output = torch.empty((N, K, OH, OW), device=input.device, dtype=input.dtype)

    num_blocks = N * triton.cdiv(K, 32) * triton.cdiv(OH, 16) * triton.cdiv(OW, 16)
    grid = lambda meta: (N * triton.cdiv(K, meta['BLOCK_K'])
                         * triton.cdiv(OH, meta['BLOCK_OH'])
                         * triton.cdiv(OW, meta['BLOCK_OW']),)

    conv2d_kernel_autotuned[grid](
        input, weight, output,
        N, C, H, W, K, R, S,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    return output


if __name__ == "__main__":
    # 正确性测试
    N, C, H, W, K, R, S = 1, 16, 32, 32, 32, 3, 3
    input_t = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32)
    weight_t = torch.randn((K, C, R, S), device="cuda", dtype=torch.float32)

    print(f"=== Conv2D Correctness Test ===")
    print(f"  input:  [{N}, {C}, {H}, {W}]")
    print(f"  weight: [{K}, {C}, {R}, {S}]")

    # PyTorch reference (stride=1, pad=0)
    ref = torch.nn.functional.conv2d(input_t, weight_t, stride=1, padding=0)

    for ver in [0, 1]:
        out = conv2d(input_t, weight_t, version=ver)
        max_err = (out - ref).abs().max().item()
        passed = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
        print(f"  v{ver: <8} max_err={max_err:.2e}  {'PASS' if passed else 'FAIL'}")

    # Autotuned version
    try:
        out_at = conv2d_autotuned(input_t, weight_t)
        max_err_at = (out_at - ref).abs().max().item()
        passed_at = torch.allclose(out_at, ref, atol=1e-3, rtol=1e-3)
        print(f"  autotuned max_err={max_err_at:.2e}  {'PASS' if passed_at else 'FAIL'}")
    except Exception as e:
        print(f"  autotuned skipped: {e}")

    # Benchmark
    print(f"\n=== Conv2D Benchmark ===")
    # Warmup
    for _ in range(10):
        conv2d(input_t, weight_t, version=1)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N', 'C', 'H', 'W', 'K', 'R', 'S'],
            x_vals=[(1, 64, 56, 56, 64, 3, 3),   # mid-layer
                    (1, 128, 28, 28, 128, 3, 3),  # deep-layer
                    (1, 3, 224, 224, 64, 7, 7)],  # first-layer
            line_arg='provider',
            line_vals=['torch', 'triton_v0', 'triton_v1', 'triton_autotuned'],
            line_names=['torch.conv2d', 'Triton v0', 'Triton v1', 'Triton autotuned'],
            ylabel='GFLOPS',
            plot_name='conv2d-performance',
            args={},
        )
    )
    def bench(N, C, H, W, K, R, S, provider):
        input_t = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32)
        weight_t = torch.randn((K, C, R, S), device="cuda", dtype=torch.float32)
        OH, OW = H - R + 1, W - S + 1
        flops = 2.0 * N * K * OH * OW * C * R * S

        quantiles = [0.5, 0.2, 0.8]

        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.nn.functional.conv2d(input_t, weight_t, stride=1, padding=0),
                quantiles=quantiles)
        elif provider == 'triton_v0':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: conv2d(input_t, weight_t, version=0), quantiles=quantiles)
        elif provider == 'triton_v1':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: conv2d(input_t, weight_t, version=1), quantiles=quantiles)
        elif provider == 'triton_autotuned':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: conv2d_autotuned(input_t, weight_t), quantiles=quantiles)

        gflops = flops * 1e-9 / (ms * 1e-3)
        return gflops

    bench.run(print_data=True)
