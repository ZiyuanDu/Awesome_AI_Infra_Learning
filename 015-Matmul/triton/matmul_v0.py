import torch 
import triton 
import triton.language as tl 


@triton.jit
def matmul_kernel_v0(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)

    # 计算C在列方向上一共被切分成了多少块
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 将一维pid转换为2维坐标(pid_m, pid_n)
    pid_m = pid // num_pid_n # 当前块在第几行
    pid_n = pid % num_pid_n  # 当前块在第几列


    # 当前块具体对应的矩阵原图的行列索引
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 要进行累加的维度
    off_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算地址指针
    a_ptrs = a_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
    b_ptrs = b_ptr + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn 

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


    # 沿着K维度循环
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remain = K - k * BLOCK_SIZE_K
        a_mask = (off_m[:, None] < M) & (off_k[None, :] < k_remain)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_mask = (off_k[:, None] < k_remain) & (off_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=False)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_ptrs = c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    c_mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)



def matmul(a: torch.Tensor, b: torch.Tensor, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2, "只支持二维矩阵"
    assert a.shape[1] == b.shape[0], f"维度不匹配：{a.shape} 和 {b.shape}"
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 一维网格，总块数 = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    matmul_kernel_v0[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c


if __name__ == "__main__":
    M, K, N = 512, 256, 128
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    c_triton = matmul(a, b)
    c_torch = torch.matmul(a, b)

    print("最大误差:", (c_triton - c_torch).abs().max().item())
    print("是否通过:", torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2))