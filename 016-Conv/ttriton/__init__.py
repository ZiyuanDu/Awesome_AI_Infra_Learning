"""Triton Conv2D 教学实现:从朴素定义到 Implicit GEMM 的递进优化

v0 naive          卷积的定义,零复用(基线)
v1 spatial        空间分块 → 数据复用(最大跳变)
v2 tiled          空间+通道分块 → 提高算术强度(GEMM 雏形)
v3 im2col         重构为矩阵乘(cuBLAS)
v4 implicit_gemm  融合 im2col 的 GEMM(cuDNN 主力算法)
"""
from .conv_v0_naive import conv_v0
from .conv_v1_spatial import conv_v1
from .conv_v2_tiled import conv_v2
from .conv_v3_im2col import conv_v3
from .conv_v4_implicit_gemm import conv_v4

__all__ = ["conv_v0", "conv_v1", "conv_v2", "conv_v3", "conv_v4"]
