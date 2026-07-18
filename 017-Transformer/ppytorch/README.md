# ppytorch

按**调用层级**组织的教学版 Transformer。自底向上读，顶层只负责组装。

## 调用关系

```
transformer/model.py          # L0 整机
├── embeddings/               # 词嵌入 + 位置编码
├── blocks/encoder.py         # L1 编码器堆叠
│   └── EncoderLayer
│       ├── attention/multihead.py → attention/sdpa.py
│       └── layers/{layer_norm,feed_forward}.py
├── blocks/decoder.py         # L1 解码器堆叠 (多交叉注意力)
└── attention/masking.py      # 因果 / 填充掩码
```

| 层级 | 目录 | 职责 |
|------|------|------|
| L3 | `attention/sdpa.py`, `masking.py` | 注意力公式与掩码 |
| L3 | `layers/` | LayerNorm / FFN |
| L3 | `embeddings/` | Token / 位置编码 |
| L2 | `attention/multihead.py` | 多头组装 |
| L1 | `blocks/` | Encoder / Decoder |
| L0 | `transformer/` | 整机 + generate |

## 设计约定

- **`batch_first`**: `[B, L, d_model]`
- **加性掩码**: `0` 保留、`-inf` 屏蔽，可广播到 `[B, h, Lq, Lk]`
- **`pe_type`**: `"sinusoidal"`（默认）或 `"learned"`

## 推荐阅读顺序

1. `attention/sdpa.py` → `attention/masking.py` → `attention/multihead.py`
2. `layers/` → `embeddings/`
3. `blocks/encoder.py` → `blocks/decoder.py`
4. `transformer/model.py` → `verify.py`

## 快速开始

在 `017-Transformer/`（包的上一级）运行:

```bash
# 单模块对拍 (在 017-Transformer/ 下)
python3 -m ppytorch.attention.sdpa
python3 -m ppytorch.attention.multihead
python3 -m ppytorch.blocks.encoder

# 全部对拍 + 复制任务
python3 -m ppytorch.verify
# 或
python3 -m ppytorch
```

依赖: `torch>=2.0`, `rich`。
