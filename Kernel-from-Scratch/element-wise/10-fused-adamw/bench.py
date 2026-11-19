from math import sqrt
from pathlib import Path
from statistics import median

import torch
from torch.utils.cpp_extension import load_inline


cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor params, torch::Tensor grad, torch::Tensor m, torch::Tensor v,
         float lr, float beta1, float beta2, float eps, float decay, int t) {
    solve(params.data_ptr<float>(), grad.data_ptr<float>(), m.data_ptr<float>(),
          v.data_ptr<float>(), params.numel(), lr, beta1, beta2, eps, decay, t);
}
"""
custom = load_inline(
    name="custom_fused_adamw",
    cpp_sources="""void run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                            float, float, float, float, float, int);""",
    cuda_sources=cuda,
    functions="run",
    extra_cuda_cflags=["-O3"],
).run


def manual_step(params, grad, m, v, lr, beta1, beta2, eps, decay, t):
    params.mul_(1 - lr * decay)
    m.lerp_(grad, 1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    sqrt_bias2 = sqrt(1 - beta2**t)
    denom = v.sqrt().add_(eps * sqrt_bias2)
    params.addcdiv_(m, denom, value=-lr * sqrt_bias2 / (1 - beta1**t))


def time_step(step, reset, warmup=3, repeats=15):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(warmup + repeats):
        reset()
        start.record()
        step()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return median(times[warmup:]) * 1000


if __name__ == "__main__":
    torch.manual_seed(42)
    n, t = 2**24, 1000
    lr, beta1, beta2, eps, decay = 0.1, 0.9, 0.999, 1e-8, 0.01
    args = (lr, beta1, beta2, eps, decay, t)

    params, grad, m, v = (
        torch.empty(n, device="cuda").uniform_(lo, hi)
        for lo, hi in ((-2, 2), (-1, 1), (-0.5, 0.5), (0.01, 0.5))
    )
    state = (params, m, v)
    initial = tuple(x.clone() for x in state)

    parameter = torch.nn.Parameter(params)
    parameter.grad = grad
    optimizer = torch.optim.AdamW(
        [parameter], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=decay, fused=True
    )
    step_count = torch.tensor(float(t - 1), device="cuda")
    optimizer.state[parameter] = dict(step=step_count, exp_avg=m, exp_avg_sq=v)
    optimizer.step()
    expected = tuple(x.clone() for x in state)

    def reset():
        for target, source in zip(state, initial):
            target.copy_(source)
        step_count.fill_(t - 1)

    methods = {
        "CUDA": lambda: custom(params, grad, m, v, *args),
        "PyTorch fused": optimizer.step,
        "PyTorch manual": lambda: manual_step(params, grad, m, v, *args),
    }
    print(f"{torch.cuda.get_device_name()}  N={n:,}  t={t:,}")
    for name, step in methods.items():
        reset()
        step()
        torch.testing.assert_close(state, expected, rtol=1e-4, atol=1e-5)
        print(f"{name:>14}  {time_step(step, reset):8.2f} us")
