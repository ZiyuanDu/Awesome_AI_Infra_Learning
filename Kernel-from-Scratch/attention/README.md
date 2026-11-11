# Attention / 注意力

从单头前向到因果、多头共享、单 query 解码和分页 KV。

每题包含 `README.md`、`solve.cu` 和 `bench.py`。前置练习见各题题面；专项题可按目标岗位选做。

| 题目 | 目录 | 层级 |
|---|---|---|
| Softmax Attention / Naive | [01-naive](01-naive/README.md) | 基础 |
| Softmax Attention / Online | [02-online](02-online/README.md) | 基础 |
| Softmax Attention / Flash v2 | [03-flash-v2](03-flash-v2/README.md) | 基础 |
| Causal Self-Attention / 因果自注意力 | [04-causal-attention](04-causal-attention/README.md) | 专项 |
| GQA / 分组查询注意力 | [05-grouped-query-attention](05-grouped-query-attention/README.md) | 专项 |
| Decode Attention / 连续 KV Cache 解码 | [06-decode-attention](06-decode-attention/README.md) | 专项 |
| Paged KV-Cache Attention / 分页解码 | [07-paged-kv-attention](07-paged-kv-attention/README.md) | 专项 |
| Split-KV Reduction / 合并注意力分片 | [08-split-kv-reduction](08-split-kv-reduction/README.md) | 融合 |
