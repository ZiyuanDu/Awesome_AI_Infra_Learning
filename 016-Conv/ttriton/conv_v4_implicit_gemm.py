import torch, triton, triton.language as tl
from .common import (parse_conv_params, compute_output_dims, pad_input,
                     nchw_from_gemm, assert_close)


@triton.autotune(
    configs=[triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
                           num_warps=nw, num_stages=ns)
             for bm, bn, bk, nw, ns in [
                 (32,  64, 32, 4, 3),
                 (64,  64, 32, 4, 3),
                 (64, 128, 32, 8, 3),
                 (128, 64, 32, 8, 3),
                 (64, 128, 64, 8, 4),
                 (128, 128, 32, 8, 4),
                 (32, 128, 32, 4, 3),
                 (64,  64, 16, 4, 2),
             ]],
    key=['C_in', 'C_out', 'kH', 'kW', 'H_out', 'W_out'],
)
@triton.jit
def _kernel(input_ptr, weight_ptr, output_ptr, bias_ptr,
            B, C_in, H_in, W_in, C_out, kH, kW,
            sh, sw, H_out, W_out,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < C_out

    N_total = B * H_out * W_out
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N_total

    hw = H_out * W_out
    b_idx = n_offs // hw
    rem = n_offs % hw
    h_in = (rem // W_out) * sh
    w_in = (rem % W_out) * sw

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    CRS = C_in * kH * kW
    khw = kH * kW

    for k_start in range(0, CRS, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < CRS

        ci = k_offs // khw
        rs = k_offs % khw
        kh_ = rs // kW
        kw_ = rs % kW

        wgt = tl.load(weight_ptr + m_offs[:, None] * CRS + k_offs[None, :],
                      mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        in_h = h_in[None, :] + kh_[:, None]
        in_w = w_in[None, :] + kw_[:, None]
        in_idx = (b_idx[None, :] * C_in * H_in * W_in
                  + ci[:, None] * H_in * W_in
                  + in_h * W_in + in_w)
        inp = tl.load(input_ptr + in_idx,
                      mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(wgt, inp)

    if bias_ptr is not None:
        acc += tl.load(bias_ptr + m_offs, mask=m_mask, other=0.0)[:, None]

    tl.store(output_ptr + m_offs[:, None] * N_total + n_offs[None, :],
             acc, mask=m_mask[:, None] & n_mask[None, :])


def conv_v4(x, w, bias=None, stride=1, padding=0):
    """Implicit GEMM 卷积 (autotuned)"""
    B, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = w.shape
    sh, sw, ph, pw = parse_conv_params(stride, padding)

    x_pad = pad_input(x, ph, pw)
    _, _, H_pad, W_pad = x_pad.shape
    H_out, W_out = compute_output_dims(H_in, W_in, kH, kW, sh, sw, ph, pw)

    N_total = B * H_out * W_out
    out = torch.empty(C_out, N_total, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(C_out, meta['BLOCK_M']),
                         triton.cdiv(N_total, meta['BLOCK_N']))
    _kernel[grid](x_pad, w, out, bias if bias is not None else None,
                  B, C_in, H_pad, W_pad, C_out, kH, kW, sh, sw, H_out, W_out)
    return nchw_from_gemm(out, B, C_out, H_out, W_out)


if __name__ == "__main__":
    for B, C_in, H, W, C_out, kH, kW, pad in [
        (1, 3, 32, 32, 16, 3, 3, 1),
        (1, 64, 56, 56, 64, 3, 3, 0),
        (1, 3, 56, 56, 64, 7, 7, 3),
        (4, 32, 16, 16, 64, 3, 3, 1),
    ]:
        x = torch.randn(B, C_in, H, W, device='cuda')
        w = torch.randn(C_out, C_in, kH, kW, device='cuda')
        b = torch.randn(C_out, device='cuda')
        out = conv_v4(x, w, b, padding=pad)
        ref = torch.nn.functional.conv2d(x, w, b, padding=pad)
        assert_close(out, ref, name=f"v4 [{B},{C_in},{H},{W}]→[{C_out},{kH},{kW}]")
    print("✅ v4 Implicit GEMM (autotuned) 全部验证通过")
