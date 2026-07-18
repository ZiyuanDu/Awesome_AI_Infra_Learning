"""
统一验证 + 端到端 demo (包入口)。

  1) 按调用层级自底向上对各模块 main() 对拍
  2) 复制任务端到端训练
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppytorch.console import console, render_summary, samples_table, section, train_table


def run_module_checks():
    section("第一部分: 各模块 vs 官方 PyTorch 对拍 (自底向上)")

    from ppytorch.attention import masking, multihead, sdpa
    from ppytorch.blocks import decoder, encoder
    from ppytorch.embeddings import positional, token
    from ppytorch.layers import feed_forward, layer_norm
    from ppytorch.transformer import model as transformer_model

    results = [
        # L3 底层
        sdpa.main(),
        masking.main(),
        layer_norm.main(),
        feed_forward.main(),
        token.main(),
        positional.main(),
        # L2 注意力组装
        multihead.main(),
        # L1 编码器 / 解码器
        encoder.main(),
        decoder.main(),
        # L0 整机
        transformer_model.main(),
    ]
    return render_summary(results)


def train_copy_task(steps=300, device=None):
    from ppytorch.transformer import Transformer

    section("第二部分: 复制任务端到端训练 (整机链路验证)")
    console.print("  [dim]任务: 输入一串随机 token, 要求原样输出。"
                  "几百步内可在 CPU 收敛。[/]")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    V = 20
    PAD, BOS, EOS = 0, 1, 2
    d_model, seq = 32, 8

    model = Transformer(
        V, V, d_model=d_model, num_heads=4,
        num_encoder_layers=2, num_decoder_layers=2,
        d_ff=64, dropout=0.0, pad_idx=PAD,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    def make_batch(B=32):
        content = torch.randint(3, V, (B, seq), device=device)
        bos_col = torch.full((B, 1), BOS, device=device)
        eos_col = torch.full((B, 1), EOS, device=device)
        seqs = torch.cat([bos_col, content, eos_col], dim=1)
        return seqs, seqs

    history = []
    model.train()
    for step in range(steps):
        src, tgt = make_batch()
        logits = model(src, tgt[:, :-1])
        loss = loss_fn(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append((step, loss.item()))

    train_table(history)

    model.eval()
    src, _ = make_batch(B=3)
    gen = model.generate(src, bos_idx=BOS, eos_idx=EOS, max_len=seq + 2)

    correct = total = 0
    rows = []
    for i in range(src.size(0)):
        ref = src[i, 1:-1].tolist()
        out = [t for t in gen[i, 1:].tolist() if t not in (EOS, PAD)][:len(ref)]
        correct += sum(a == b for a, b in zip(ref, out))
        total += len(ref)
        rows.append((i, str(ref), str(out), all(a == b for a, b in zip(ref, out))))

    samples_table(rows)
    acc_style = "bold bright_green" if correct / total > 0.85 else "bright_yellow"
    console.print(f"  复制准确率: [{acc_style}]{correct}/{total} = {correct / total:.1%}[/]")


def main():
    run_module_checks()
    train_copy_task()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
