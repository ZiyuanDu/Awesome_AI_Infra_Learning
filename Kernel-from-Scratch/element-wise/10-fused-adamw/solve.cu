#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 4096;

struct AdamWStep {
    float beta1;
    float beta2;
    float decay;
    float eps;
    double step_size;
};

__device__ __forceinline__ void adamw(float& param, float grad, float& m, float& v,
                                       const AdamWStep& step) {
    param *= step.decay;
    m = step.beta1 * m + (1.0f - step.beta1) * grad;
    v = step.beta2 * v + (1.0f - step.beta2) * (grad * grad);

    const float denom = sqrtf(v) + step.eps;
    const float step_size = static_cast<float>(step.step_size);
    const float numerator = step_size * m;
    const float magnitude = fabsf(numerator);
    // 极端输入用 FP64 计算步长，避免中间乘积提前下溢或溢出。
    if (step_size < FLT_MIN || magnitude > FLT_MAX || (m != 0.0f && magnitude < FLT_MIN)) {
        param -= static_cast<float>(step.step_size * m / denom);
    } else {
        param -= numerator / denom;
    }
}

__device__ __forceinline__ void adamw4(float4& param, float4 grad, float4& m, float4& v,
                                        const AdamWStep& step) {
    adamw(param.x, grad.x, m.x, v.x, step);
    adamw(param.y, grad.y, m.y, v.y, step);
    adamw(param.z, grad.z, m.z, v.z, step);
    adamw(param.w, grad.w, m.w, v.w, step);
}

__global__ void adamw_kernel(float* __restrict__ params, const float* __restrict__ grad,
                             float* __restrict__ m, float* __restrict__ v,
                             int N, int N4, AdamWStep step) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    auto* params4 = reinterpret_cast<float4*>(params);
    const auto* grad4 = reinterpret_cast<const float4*>(grad);
    auto* m4 = reinterpret_cast<float4*>(m);
    auto* v4 = reinterpret_cast<float4*>(v);
    for (int i = tid; i < N4; i += stride) {
        float4 param = params4[i];
        float4 first = m4[i];
        float4 second = v4[i];
        adamw4(param, grad4[i], first, second, step);
        params4[i] = param;
        m4[i] = first;
        v4[i] = second;
    }

    for (int i = 4 * N4 + tid; i < N; i += stride) {
        adamw(params[i], grad[i], m[i], v[i], step);
    }
}

void solve(
    float* params,
    const float* grad,
    float* m,
    float* v,
    int N,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t
) {
    if (N <= 0) return;

    // 提前做偏差校正，将 sqrt(1 - beta2^t) 移到分子和 eps 中。
    const double bias1 = 1.0 - std::pow(static_cast<double>(beta1), t);
    const double sqrt_bias2 = std::sqrt(1.0 - std::pow(static_cast<double>(beta2), t));
    const AdamWStep step{beta1, beta2, 1.0f - lr * weight_decay,
                         static_cast<float>(eps * sqrt_bias2), lr * sqrt_bias2 / bias1};

    const auto addresses = reinterpret_cast<std::uintptr_t>(params) |
                           reinterpret_cast<std::uintptr_t>(grad) |
                           reinterpret_cast<std::uintptr_t>(m) |
                           reinterpret_cast<std::uintptr_t>(v);
    const int N4 = addresses % alignof(float4) == 0 ? N / 4 : 0;
    const int blocks = std::min(CEIL(N4 > 0 ? N4 : N, THREADS), MAX_BLOCKS);
    adamw_kernel<<<blocks, THREADS>>>(params, grad, m, v, N, N4, step);
}
