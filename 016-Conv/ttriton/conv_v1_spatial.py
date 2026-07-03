import torch, triton, triton.language as tl
from .common import parse_conv_params, compute_output_dims, pad_input, assert_close


@triton.autotune(
    configs=[triton.Config({'TILE_H': th, 'TILE_W': tw}, num_warps=nw)
             for th in [4, 8, 16] for tw in [8, 16, 32]
             for nw in [2, 4]
             if 32 <= th * tw <= 256],
    key=['C_in', 'C_out', 'kH', 'kW'],
)
@triton.jit
def _kernel(input_ptr, weight_ptr, output_ptr, bias_ptr,
            B, C_in, H_in, W_in, C_out, kH, kW,
            sh, sw, H_out, W_out,
            TILE_H: tl.constexpr, TILE_W: tl.constexpr):

    pid_hw = tl.program_id(0)
    co = tl.program_id(1)
    b = tl.program_id(2)

    n_tiles_w = tl.cdiv(W_out, TILE_W)
    tile_h = pid_hw // n_tiles_w
    tile_w = pid_hw % n_tiles_w

    h_offs = tile_h * TILE_H + tl.arange(0, TILE_H)
    w_offs = tile_w * TILE_W + tl.arange(0, TILE_W)
    out_mask = (h_offs[:, None] < H_out) & (w_offs[None, :] < W_out)

    acc = tl.zeros([TILE_H, TILE_W], dtype=tl.float32)
    in_b_base = b * C_in * H_in * W_in
    wgt_co_base = co * C_in * kH * kW

    for ci in range(C_in):
        in_ci_base = in_b_base + ci * H_in * W_in
        wgt_ci_base = wgt_co_base + ci * kH * kW
        for kh in range(kH):
            for kw in range(kW):
                in_h = h_offs * sh + kh
                in_w = w_offs * sw + kw

                in_load_mask = (in_h[:, None] < H_in) & (in_w[None, :] < W_in)
                in_vals = tl.load(
                    input_ptr + in_ci_base + in_h[:, None] * W_in + in_w[None, :],
                    mask=in_load_mask, other=0.0)
                wgt_val = tl.load(weight_ptr + wgt_ci_base + kh * kW + kw)
                acc += in_vals * wgt_val

    if bias_ptr is not None:
        acc += tl.load(bias_ptr + co)

    out_base = b * C_out * H_out * W_out + co * H_out * W_out
    out_idx = out_base + h_offs[:, None] * W_out + w_offs[None, :]
    tl.store(output_ptr + out_idx, acc, mask=out_mask)


def conv_v1(x, w, bias=None, stride=1, padding=0):
    B, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = w.shape
    sh, sw, ph, pw = parse_conv_params(stride, padding)

    x_pad = pad_input(x, ph, pw)
    _, _, H_pad, W_pad = x_pad.shape
    H_out, W_out = compute_output_dims(H_in, W_in, kH, kW, sh, sw, ph, pw)

    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(H_out, meta['TILE_H'])
                         * triton.cdiv(W_out, meta['TILE_W']),
                         C_out, B)
    _kernel[grid](x_pad, w, out, bias if bias is not None else None,
                  B, C_in, H_pad, W_pad, C_out, kH, kW, sh, sw, H_out, W_out)
    return out


if __name__ == "__main__":
    x = torch.randn(1, 64, 56, 56, device='cuda')
    w = torch.randn(64, 64, 3, 3, device='cuda')
    b = torch.randn(64, device='cuda')
    out = conv_v1(x, w, b)
    ref = torch.nn.functional.conv2d(x, w, b)
    assert_close(out, ref, name="v1")
    print("✅ v1 空间分块 (autotuned) 验证通过")
