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
    name="custom_sigmoid",
    cpp_sources="void run(torch::Tensor, torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
).run


if __name__ == "__main__":
    a = torch.rand(2**10+3, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)

    custom(a, out)
    torch.testing.assert_close(out, torch.nn.functional.sigmoid(a))
    print("test successfully!")

    for stmt in("custom(a, out)", "torch.nn.functional.sigmoid(a)"):
        us = Timer(stmt=stmt, globals=globals()).blocked_autorange().median * 1e6
        print(f"{stmt}: {us:.3f} us")
