import torch, triton, triton.language as tl
from .common import parse_conv_params, compute_output_dims, pad_input, assert_close


@triton.autotune(
    configs=[triton.Config({'TILE_H': th, 'TILE_W': tw, 'BLOCK_K': bk}, num_warps=nw)
             for th in [4, 8] for tw in [8, 16] for bk in [16, 32, 64]
             for nw in [4, 8]
             if th * tw * bk <= 4096],
    key=['C_in', 'C_out', 'kH', 'kW'],
)
@triton.jit
def _kernel(input_ptr, weight_ptr, output_ptr, bias_ptr,
            B, C_in, H_in, W_in, C_out, kH, kW,
            sh, sw, H_out, W_out,
            TILE_H: tl.constexpr, TILE_W: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_hw = tl.program_id(0)
    pid_k = tl.program_id(1)
    b = tl.program_id(2)

    n_tiles_w = tl.cdiv(W_out, TILE_W)
    tile_h = pid_hw // n_tiles_w
    tile_w = pid_hw % n_tiles_w

    h_offs = tile_h * TILE_H + tl.arange(0, TILE_H)
    w_offs = tile_w * TILE_W + tl.arange(0, TILE_W)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < C_out

    acc = tl.zeros([TILE_H, TILE_W, BLOCK_K], dtype=tl.float32)
    wgt_stride_k = C_in * kH * kW

    for ci in range(C_in):
        in_ci_base = b * C_in * H_in * W_in + ci * H_in * W_in
        wgt_ci_base = ci * kH * kW
        for kh in range(kH):
            for kw in range(kW):
                in_h = h_offs * sh + kh
                in_w = w_offs * sw + kw
                in_load_mask = (in_h[:, None] < H_in) & (in_w[None, :] < W_in)

                in_2d = tl.load(
                    input_ptr + in_ci_base + in_h[:, None] * W_in + in_w[None, :],
                    mask=in_load_mask, other=0.0)

                wgt_idx = k_offs * wgt_stride_k + wgt_ci_base + kh * kW + kw
                wgt_1d = tl.load(weight_ptr + wgt_idx, mask=k_mask, other=0.0)

                acc += in_2d[:, :, None] * wgt_1d[None, None, :]

    if bias_ptr is not None:
        acc += tl.load(bias_ptr + k_offs, mask=k_mask, other=0.0)[None, None, :]

    out_idx = (b * C_out * H_out * W_out
               + k_offs[None, None, :] * H_out * W_out
               + h_offs[:, None, None] * W_out
               + w_offs[None, :, None])
    out_mask = ((h_offs[:, None, None] < H_out)
                & (w_offs[None, :, None] < W_out)
                & k_mask[None, None, :])
    tl.store(output_ptr + out_idx, acc, mask=out_mask)


def conv_v2(x, w, bias=None, stride=1, padding=0):
    B, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = w.shape
    sh, sw, ph, pw = parse_conv_params(stride, padding)

    x_pad = pad_input(x, ph, pw)
    _, _, H_pad, W_pad = x_pad.shape
    H_out, W_out = compute_output_dims(H_in, W_in, kH, kW, sh, sw, ph, pw)

    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(H_out, meta['TILE_H'])
                         * triton.cdiv(W_out, meta['TILE_W']),
                         triton.cdiv(C_out, meta['BLOCK_K']), B)
    _kernel[grid](x_pad, w, out, bias if bias is not None else None,
                  B, C_in, H_pad, W_pad, C_out, kH, kW, sh, sw, H_out, W_out)
    return out


if __name__ == "__main__":
    x = torch.randn(1, 64, 56, 56, device='cuda')
    w = torch.randn(64, 64, 3, 3, device='cuda')
    b = torch.randn(64, device='cuda')
    out = conv_v2(x, w, b)
    ref = torch.nn.functional.conv2d(x, w, b)
    assert_close(out, ref, name="v2")
    print("空间+通道分块验证通过")
