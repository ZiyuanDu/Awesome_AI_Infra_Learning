"""
整机 Transformer — 调用层级最顶层。

  src → Embedding + PE → Encoder → memory
  tgt → Embedding + PE → Decoder(memory, masks) → logits

pe_type: "sinusoidal" | "learned"
generate: 贪心自回归 (教学版每步重算 decoder, 无 KV cache)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppytorch.attention import causal_mask, combine_masks, padding_mask
from ppytorch.blocks import Decoder, Encoder
from ppytorch.embeddings import LearnedPE, SinusoidalPE, TokenEmbedding


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 dropout=0.1, activation="relu", norm_first=False,
                 pad_idx=0, tie_weights=True, pe_type="sinusoidal",
                 max_len=5000):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_emb = TokenEmbedding(src_vocab, d_model, padding_idx=pad_idx)
        self.tgt_emb = TokenEmbedding(tgt_vocab, d_model, padding_idx=pad_idx)

        if pe_type == "sinusoidal":
            self.pos_enc = SinusoidalPE(d_model, max_len=max_len)
        elif pe_type == "learned":
            self.pos_enc = LearnedPE(d_model, max_len=max_len)
        else:
            raise ValueError(f"pe_type 须为 'sinusoidal' 或 'learned', 得到 {pe_type!r}")

        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout, activation, norm_first)
        self.decoder = Decoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout, activation, norm_first)
        self.output_proj = nn.Linear(d_model, tgt_vocab, bias=False)
        if tie_weights:
            self.output_proj.weight = self.tgt_emb.weight
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids, src_pad_mask=None):
        x = self.dropout(self.pos_enc(self.src_emb(src_ids)))
        return self.encoder(x, src_pad_mask)

    def decode(self, tgt_ids, memory, tgt_mask=None, memory_mask=None):
        x = self.dropout(self.pos_enc(self.tgt_emb(tgt_ids)))
        return self.decoder(x, memory, tgt_mask, memory_mask)

    def forward(self, src_ids, tgt_ids):
        """训练前向 (teacher forcing)。→ logits [B, Lt, V]"""
        Lt = tgt_ids.size(1)
        src_pad = padding_mask(src_ids == self.pad_idx)
        tgt_pad = padding_mask(tgt_ids == self.pad_idx)
        tgt_mask = combine_masks(causal_mask(Lt, device=src_ids.device), tgt_pad)

        memory = self.encode(src_ids, src_pad)
        dec = self.decode(tgt_ids, memory, tgt_mask, src_pad)
        return self.output_proj(dec)

    @torch.no_grad()
    def generate(self, src_ids, bos_idx, eos_idx, max_len=50):
        """贪心自回归生成。→ [B, <=max_len]"""
        self.eval()
        src_pad = padding_mask(src_ids == self.pad_idx)
        memory = self.encode(src_ids, src_pad)
        ys = torch.full((src_ids.size(0), 1), bos_idx, dtype=torch.long, device=src_ids.device)
        for _ in range(max_len - 1):
            tgt_mask = causal_mask(ys.size(1), device=src_ids.device)
            logits = self.output_proj(self.decode(ys, memory, tgt_mask, src_pad)[:, -1])
            next_tok = logits.argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_idx).all():
                break
        return ys


def main():
    torch.manual_seed(0)
    model = Transformer(
        src_vocab=50, tgt_vocab=50, d_model=32, num_heads=4,
        num_encoder_layers=2, num_decoder_layers=2, d_ff=64, dropout=0.0)
    src = torch.randint(1, 50, (2, 6))
    tgt = torch.randint(1, 50, (2, 4))
    logits = model(src, tgt)
    gen = model.generate(src, bos_idx=1, eos_idx=2, max_len=10)
    n_params = sum(p.numel() for p in model.parameters())

    # learned PE 路径也可跑通
    model_lp = Transformer(
        50, 50, d_model=32, num_heads=4, num_encoder_layers=1, num_decoder_layers=1,
        d_ff=64, dropout=0.0, pe_type="learned", max_len=32)
    logits_lp = model_lp(src, tgt)

    from ppytorch.console import ModuleResult
    return ModuleResult(
        "transformer.model", "整机前向 + 贪心生成",
        info=[("logits", tuple(logits.shape)),
              ("generate", tuple(gen.shape)),
              ("参数量", f"{n_params:,}")],
        checks=[
            ("logits 形状 [B,Lt,V]", tuple(logits.shape) == (2, 4, 50)),
            ("生成含 bos", gen[0, 0].item() == 1),
            ("pe_type=learned 可前向", tuple(logits_lp.shape) == (2, 4, 50)),
        ],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())
