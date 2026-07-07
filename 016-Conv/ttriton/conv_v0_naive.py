import torch, triton, triton.language as tl
from .common import parse_conv_params, compute_output_dims, pad_input, assert_close


@triton.jit
def _kernel(input_ptr, weight_ptr, output_ptr, bias_ptr,
            B, C_in, H_in, W_in, C_out, kH, kW,
            sh, sw, H_out, W_out):

    pid = tl.program_id(0)
    w = pid % W_out; rem = pid // W_out
    h = rem % H_out; rem = rem // H_out
    co = rem % C_out; b = rem // C_out

    h_in = h * sh; w_in = w * sw
    acc = tl.zeros([], dtype=tl.float32)

    for ci in range(C_in):
        in_c_base = b * C_in * H_in * W_in + ci * H_in * W_in
        wgt_c_base = co * C_in * kH * kW + ci * kH * kW
        for kh in range(kH):
            for kw in range(kW):
                in_idx = in_c_base + (h_in + kh) * W_in + (w_in + kw)
                wgt_idx = wgt_c_base + kh * kW + kw
                acc += tl.load(input_ptr + in_idx) * tl.load(weight_ptr + wgt_idx)

    if bias_ptr is not None:
        acc += tl.load(bias_ptr + co)

    out_idx = b * C_out * H_out * W_out + co * H_out * W_out + h * W_out + w
    tl.store(output_ptr + out_idx, acc)


def conv_v0(x, w, bias=None, stride=1, padding=0):
    B, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = w.shape
    sh, sw, ph, pw = parse_conv_params(stride, padding)

    x_pad = pad_input(x, ph, pw)
    _, _, H_pad, W_pad = x_pad.shape
    H_out, W_out = compute_output_dims(H_in, W_in, kH, kW, sh, sw, ph, pw)

    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    grid = (B * C_out * H_out * W_out,)
    _kernel[grid](x_pad, w, out, bias if bias is not None else None,
                  B, C_in, H_pad, W_pad, C_out, kH, kW, sh, sw, H_out, W_out)
    return out


if __name__ == "__main__":
    x = torch.randn(1, 3, 16, 16, device='cuda')
    w = torch.randn(8, 3, 3, 3, device='cuda')
    b = torch.randn(8, device='cuda')
    out = conv_v0(x, w, b, padding=1)
    ref = torch.nn.functional.conv2d(x, w, b, padding=1)
    assert_close(out, ref, name="v0")
    print("朴素卷积验证通过")
