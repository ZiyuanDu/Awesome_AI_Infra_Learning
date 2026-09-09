# GEMM

| 题目 | 目录 | 要点 |
|---|---|---|
| Matrix Mul | `01_matrix_mul` | FP32；\(A_{M\times N} B_{N\times K}\) |
| SpMV | `02_spmv` | 稠密布局 GEMV；x→smem + float4 |
| GEMM | `03_gemm` | half I/O + FP32 累加；WMMA + cp.async；见 `题目/23-GEMM.md` |
