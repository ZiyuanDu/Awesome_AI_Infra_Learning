# CUDA Kernel Learning

CUDA 算子学习仓库，包含参考实现、从零练习和技术笔记。

| 项目 | 内容 | 适合用途 |
|---|---|---|
| [Kernel-Reference](Kernel-Reference/README.md) | 27 个题目 / 版本的 CUDA 实现、C++ 基准测试和共用工具 | 阅读参考答案、复习并行写法 |
| [Kernel-from-Scratch](Kernel-from-Scratch/README.md) | 29 个题目 / 版本的题面、CUDA 练习代码和 Python 测试入口 | 独立实现，再对照参考答案 |

阅读顺序：Element-wise → Reduce / Scan → Permute → GEMM → Stencil → Attention → Sort。

## 博客与笔记

- [按并行写法分类的题单](Kernel-Reference/题目/00-分类.md)
- [CUDA Warp Shuffle 教程](Kernel-Reference/科普/01-shuffle.md)
- [题目说明](Kernel-Reference/题目/)

## 提交与推送

在仓库根目录执行：

```bash
git add -A
git diff --cached --stat
git commit -m "更新算子代码和笔记"
git push origin main
```

`.gitignore` 默认只放行源码、Markdown 和必要的构建 / 依赖配置。博客配图放在各项目的 `assets/` 目录，支持 PNG、JPEG、GIF、WebP 和 SVG。
构建目录、虚拟环境、编辑器配置、数据、模型、性能输出和 PDF 课件留在本地。
