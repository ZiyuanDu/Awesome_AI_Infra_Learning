from pathlib import Path
from tempfile import gettempdir
import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline

folder = Path(__file__).resolve().parent
build = Path(gettempdir()) / "interview_softmax"
build.mkdir(exist_ok=True)
cuda = (folder / "solve.cu").read_text() + r"""
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
void run(torch::Tensor x, torch::Tensor y) {
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream() == c10::cuda::getDefaultCUDAStream(),
                "this bench uses the default CUDA stream");
    solve(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
}
"""
custom = load_inline(
    name="interview_softmax", 
    cpp_sources="void run(torch::Tensor, torch::Tensor);",
    cuda_sources=cuda, 
    functions=["run"], 
    extra_include_paths=[str(folder)],
    extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], 
    build_directory=str(build),
).run

if __name__ == "__main__":
    torch.manual_seed(0)
    sizes = sorted({1, 3, 500000} | {(1 << p) + d for p in range(5, 19) for d in (-1, 0, 1)})
    for N in sizes:
        for offset in range(4):
            x = torch.randn(N + 3, device="cuda")[offset:offset+N]
            expected = torch.softmax(x.double(), 0).float()
            for shift in range(4):
                storage = torch.full((N + 6,), -1234.0, device="cuda")
                y = storage[shift:shift+N]
                custom(x, y)
                torch.testing.assert_close(y, expected, rtol=3e-5, atol=2e-12)
                assert (storage[:shift] == -1234).all() and (storage[shift+N:] == -1234).all()
    print(f"Correctness: {16*len(sizes)} checks passed; {torch.cuda.get_device_name()}")
    print(f"{'N':>8} {'CUDA us':>10} {'Torch us':>10} {'speedup':>9}")
    for N in [32, 1024, 4096, 8192, 16384, 16385, 65536, 500000]:
        x = torch.empty(N, device="cuda").uniform_(-10, 10)
        y = torch.empty_like(x)
        methods = {"cuda": lambda: custom(x, y), "torch": lambda: torch.softmax(x, 0, out=y)}
        times = {name: Timer("fn()", globals={"fn": fn}).blocked_autorange(min_run_time=0.2).median * 1e6
                 for name, fn in methods.items()}
        print(f"{N:8d} {times['cuda']:10.3f} {times['torch']:10.3f} {times['torch']/times['cuda']:8.2f}x")
