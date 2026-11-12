from pathlib import Path
import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline


cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor input, torch::Tensor output) {
    solve(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
}
"""

custom = load_inline(
    name="custom_swiglu",
    cpp_sources="void run(torch::Tensor, torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
).run

def swiglu(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(x1) * x2 

if __name__ == "__main__":
    a = torch.rand(2**10, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a[:a.numel() // 2])

    custom(a, out)
    torch.testing.assert_close(out, swiglu(a))
    print("test successfully!")

    for stmt in("custom(a, out)", "swiglu(a)"):
        us = Timer(stmt=stmt, globals=globals()).blocked_autorange().median * 1e6
        print(f"{stmt}: {us:.3f} us")
