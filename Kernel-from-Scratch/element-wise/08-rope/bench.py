from pathlib import Path
from tempfile import gettempdir

import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline

cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor q, torch::Tensor cos, torch::Tensor sin, torch::Tensor out) {
    solve(q.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(), 
        out.data_ptr<float>(), q.size(0), q.size(1));
}
"""

build = Path(gettempdir()) / "custom_rope" 
build.mkdir(exist_ok=True)
custom = load_inline(
    name="custom_rope",
    cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    build_directory=str(build),
).run 

def rope(q, cos, sin, out=None):
    a, b = q.chunk(2, dim=-1)
    return torch.add(q*cos, torch.cat((-b, a), dim=-1) * sin, out=out)


if __name__ == "__main__":
    torch.manual_seed(42)

    print(torch.cuda.get_device_name())
    (M, D) = (3, 6)
    angles = torch.randn(M, D//2, device="cuda")
    q = torch.randn(M, D, device="cuda")

    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    out = torch.empty_like(q) 
    expected = rope(q, cos, sin)

    custom(q, cos, sin, out)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    error = (out - expected).abs().max().item()
    
    methods = {
        "CUDA": lambda: custom(q, cos, sin, out),
        "Torch": lambda: rope(q, cos, sin, out)
    }
    times = {
        name: Timer("fn()", globals={"fn": fn}).blocked_autorange().median * 1e6 for name, fn in methods.items()
    }
    print(f"{error:11.3e} {times['CUDA']:10.3f} {times['Torch']:10.3f} {times["Torch"] / times['CUDA']:8.2f}x")
