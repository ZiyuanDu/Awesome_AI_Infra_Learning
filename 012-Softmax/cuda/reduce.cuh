/*
 * reduce.cuh — Warp/Block 级规约原语
 *
 * 提供 Softmax 所需的通用规约操作:
 *   - WarpAllReduce:       warp 内 butterfly reduce（任意二元运算）
 *   - BlockAllReduce:      block 内 reduce（基于 CUB）
 *   - WarpAllReduceOnline: warp 内 Online softmax (m,s) 归并
 *   - BlockAllReduceOnline: block 内 Online softmax (m,s) 归并
 *
 * 所有 kernel 共享这些基础组件。
 */

#pragma once
#include <cub/cub.cuh>
#include <math_constants.h>
#include <cuda.h>

namespace cuda_softmax {

constexpr int kWarpSize = 32;

// ---- Fast-math helpers ------------------------------------------------------
// 在 OF_SOFTMAX_USE_FAST_MATH 宏定义时使用内建快速函数

template<typename T> __inline__ __device__ T Inf();
template<> __inline__ __device__ float  Inf<float>()  { return CUDART_INF_F; }
template<> __inline__ __device__ double Inf<double>() { return CUDART_INF; }

template<typename T> __inline__ __device__ T Exp(T x);
template<> __inline__ __device__ float Exp<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __expf(x);
#else
  return exp(x);
#endif
}
template<> __inline__ __device__ double Exp<double>(double x) { return exp(x); }

template<typename T> __inline__ __device__ T Div(T a, T b);
template<> __inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}
template<> __inline__ __device__ double Div<double>(double a, double b) { return a / b; }

template<typename T> __inline__ __device__ T Log(T x);
template<> __inline__ __device__ float Log<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __logf(x);
#else
  return log(x);
#endif
}
template<> __inline__ __device__ double Log<double>(double x) { return log(x); }

// ---- 规约算子 ----------------------------------------------------------------

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

// ---- Online softmax 状态 -----------------------------------------------------

template<typename T>
struct OnlinePair {
  T m;  // running max
  T s;  // running sum (rescaled)
};

// 合并两个 Online softmax 部分状态
//   m  = max(mA, mB)
//   s  = sA * exp(mA - m) + sB * exp(mB - m)
template<typename T>
struct OnlineReductionOp {
  __device__ __forceinline__ OnlinePair<T> operator()(const OnlinePair<T>& a,
                                                       const OnlinePair<T>& b) const {
    T m = max(a.m, b.m);
    T s = a.s * Exp(a.m - m) + b.s * Exp(b.m - m);
    return {m, s};
  }
};

// ---- Warp 级 all-reduce (butterfly shuffle) ----------------------------------

template<template<typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Warp 级 Online reduce: 同时归并 (m, s) pair
template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ OnlinePair<T> WarpAllReduceOnline(OnlinePair<T> val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    OnlinePair<T> other;
    other.m = __shfl_xor_sync(0xffffffff, val.m, mask);
    other.s = __shfl_xor_sync(0xffffffff, val.s, mask);
    val = OnlineReductionOp<T>()(val, other);
  }
  return val;
}

// ---- Block 级 all-reduce (via CUB) -------------------------------------------

template<template<typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) { result_broadcast = result; }
  __syncthreads();
  return result_broadcast;
}

// Block 级 Online reduce:
//   Step 1: 找全局 max
//   Step 2: rescale 每线程的 s + sum-reduce
template<typename T, int block_size>
__inline__ __device__ OnlinePair<T> BlockAllReduceOnline(OnlinePair<T> val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T broadcast_m;
  __shared__ T broadcast_s;

  // Step 1: 找全局最大值
  T m_result = BlockReduce(temp_storage).Reduce(val.m, MaxOp<T>());
  if (threadIdx.x == 0) { broadcast_m = m_result; }
  __syncthreads();
  m_result = broadcast_m;

  // Step 2: rescale + sum-reduce
  val.s *= Exp(val.m - m_result);
  val.m = m_result;
  T s_result = BlockReduce(temp_storage).Reduce(val.s, SumOp<T>());
  if (threadIdx.x == 0) { broadcast_s = s_result; }
  __syncthreads();
  s_result = broadcast_s;

  return {m_result, s_result};
}

} // namespace cuda_softmax
