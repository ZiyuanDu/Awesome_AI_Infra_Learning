import torch
import torch.nn.functional as F
from .common import parse_conv_params, compute_output_dims, nchw_from_gemm, assert_close


def conv_v3(x, w, bias=None, stride=1, padding=0):
    B, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = w.shape
    sh, sw, ph, pw = parse_conv_params(stride, padding)
    H_out, W_out = compute_output_dims(H_in, W_in, kH, kW, sh, sw, ph, pw)
    CRS = C_in * kH * kW

    col = F.unfold(x, kernel_size=(kH, kW), padding=(ph, pw), stride=(sh, sw))
    col_2d = col.permute(1, 0, 2).reshape(CRS, -1)

    out_2d = torch.matmul(w.reshape(C_out, CRS), col_2d)
    if bias is not None:
        out_2d += bias.reshape(C_out, 1)

    return nchw_from_gemm(out_2d, B, C_out, H_out, W_out)


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224, device='cuda')
    w = torch.randn(64, 3, 7, 7, device='cuda')
    b = torch.randn(64, device='cuda')
    out = conv_v3(x, w, b, padding=3)
    ref = F.conv2d(x, w, b, padding=3)
    assert_close(out, ref, name="v3")
    print("✅ v3 im2col+GEMM 验证通过")
