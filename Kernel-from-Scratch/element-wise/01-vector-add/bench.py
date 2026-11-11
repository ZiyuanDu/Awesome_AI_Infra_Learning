from pathlib import Path
import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline


cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    solve(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), A.numel());
}
"""

custom = load_inline(
    name="custom_vector_add",
    cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
).run


if __name__ == "__main__":
    a = torch.rand(2**10+3, device="cuda", dtype=torch.float32)
    b, out = torch.rand_like(a), torch.empty_like(a)

    custom(a, b, out)
    torch.testing.assert_close(out, a+b)
    print("test successfully!")

    for stmt in("custom(a, b, out)", "torch.add(a, b, out=out)"):
        us = Timer(stmt=stmt, globals=globals()).blocked_autorange().median * 1e6
        print(f"{stmt}: {us:.3f} us")
