from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CodeBERTConfig:
    model_name: str = "microsoft/codebert-base"
    max_length: int = 256
    out_dim: int = 768
    freeze: bool = True


class CodeBERTEncoder(nn.Module):
    def __init__(self, cfg: CodeBERTConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or CodeBERTConfig()
        self.cfg = cfg
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name)
        if cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.model(**enc)
        return out.last_hidden_state[:, 0, :]


def build_codebert_encoder(cfg: CodeBERTConfig | None = None) -> CodeBERTEncoder:
    return CodeBERTEncoder(cfg)
