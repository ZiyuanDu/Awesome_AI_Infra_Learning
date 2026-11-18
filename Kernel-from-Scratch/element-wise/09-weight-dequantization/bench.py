from pathlib import Path
from tempfile import gettempdir

import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline


cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor X, torch::Tensor S, torch::Tensor Y, int tile_size) {
    solve(X.data_ptr<float>(), S.data_ptr<float>(), Y.data_ptr<float>(),
          X.size(0), X.size(1), tile_size);
}
"""

build = Path(gettempdir()) / "custom_weight_dequantization"
build.mkdir(exist_ok=True)
custom = load_inline(
    name="custom_weight_dequantization",
    cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor, int);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    build_directory=str(build),
).run

if __name__ == "__main__":
    torch.manual_seed(42)

    m, n, tile_size = 8192, 8192, 128 
    x = torch.randn(m, n, device="cuda")
    s = torch.randn(m // tile_size, n // tile_size, device="cuda")
    out = torch.empty_like(x)

    tiles = x.view(s.size(0), tile_size, s.size(1), tile_size)
    out_tiles = out.view_as(tiles)

    scales = s[:, None, :, None]

    custom(x, s, out, tile_size)
    torch.testing.assert_close(out_tiles, tiles * scales)

    for name, stmt in (
        ("CUDA", "custom(x, s, out, tile_size)"),
        ("Torch", "torch.mul(tiles, scales, out=out_tiles)"),
    ):
        seconds = Timer(stmt, globals=globals()).blocked_autorange().median * 1e6
        print(f"{name}: {seconds:.2f} us")


    
