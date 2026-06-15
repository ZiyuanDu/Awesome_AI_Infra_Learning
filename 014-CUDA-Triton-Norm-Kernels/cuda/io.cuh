#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cuda_norm {

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T elem[N];
};

template <typename T, int N>
using PackType = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;

template <typename SRC, typename DST>
struct DirectLoad {
    DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}

    template <int N>
    __device__ void load(DST* dst, int64_t row, int64_t col) const {
        Pack<SRC, N> pack;
        const int64_t offset = (row * row_size + col) / N;
        pack = *reinterpret_cast<const Pack<SRC, N>*>(src + offset * N);
#pragma unroll
        for (int i = 0; i < N; ++i) dst[i] = static_cast<DST>(pack.elem[i]);
    }

    const SRC* src;
    int64_t row_size;
};

template <typename SRC, typename DST>
struct DirectStore {
    DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}

    template <int N>
    __device__ void store(const SRC* src, int64_t row, int64_t col) {
        Pack<DST, N> pack;
        const int64_t offset = (row * row_size + col) / N;
#pragma unroll
        for (int i = 0; i < N; ++i) pack.elem[i] = static_cast<DST>(src[i]);
        *reinterpret_cast<Pack<DST, N>*>(dst + offset * N) = pack;
    }

    DST* dst;
    int64_t row_size;
};

template <typename SRC, typename DST, bool do_scale, bool do_center>
struct AffineStore {
    AffineStore(DST* dst, int64_t row_size, const DST* gamma, const DST* beta)
        : dst(dst), row_size(row_size), gamma(gamma), beta(beta) {}

    template <int N>
    __device__ void store(const SRC* src, int64_t row, int64_t col) {
        Pack<DST, N> dst_pack, gamma_pack, beta_pack;
        const int64_t offset = (row * row_size + col) / N;
        const int64_t w_offset = col / N;

        if (do_scale)
            gamma_pack = *reinterpret_cast<const Pack<DST, N>*>(gamma + w_offset * N);
        if (do_center)
            beta_pack  = *reinterpret_cast<const Pack<DST, N>*>(beta  + w_offset * N);

#pragma unroll
        for (int i = 0; i < N; ++i) {
            DST v = static_cast<DST>(src[i]);
            if (do_scale)  v = v * gamma_pack.elem[i];
            if (do_center) v = v + beta_pack.elem[i];
            dst_pack.elem[i] = v;
        }
        *reinterpret_cast<Pack<DST, N>*>(dst + offset * N) = dst_pack;
    }

    DST* dst;
    int64_t row_size;
    const DST* gamma;
    const DST* beta;
};

} // namespace cuda_norm
