from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sdp.graph.builder import build_code_graph
from sdp.models.fusion.multimodal import FusionConfig, GraphOnlyClassifier, MultimodalClassifier, TextOnlyClassifier
from sdp.models.gnn.gat_encoder import GNNEncoderConfig, build_gnn_encoder
from sdp.models.text.codebert_encoder import CodeBERTConfig, build_codebert_encoder


@dataclass
class NeuralTrainConfig:
    task: str
    label_col: str = "is_bug_inducing"
    metric_cols: list[str] | None = None
    text_col: str = "diff_text"
    code_col: str = "code_text"
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-4
    seed: int = 42


class DefectDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: NeuralTrainConfig) -> None:
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.metric_cols = cfg.metric_cols or ["loc_added", "loc_deleted", "churn", "cyclomatic_complexity"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row.get(self.cfg.text_col) or row.get(self.cfg.code_col) or "")
        code = str(row.get(self.cfg.code_col) or text)
        language = str(row.get("language") or "") if "language" in row.index else None
        cpg_path = str(row.get("cpg_path") or "") if "cpg_path" in row.index else None
        graph = build_code_graph(code, language=language or None, cpg_path=cpg_path or None)
        metrics = []
        for c in self.metric_cols:
            v = row.get(c, 0.0)
            metrics.append(float(v) if pd.notna(v) else 0.0)
        y = int(row[self.cfg.label_col])
        return {
            "text": text,
            "graph": graph,
            "metrics": torch.tensor(metrics, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "texts": [b["text"] for b in batch],
        "graphs": [b["graph"] for b in batch],
        "metrics": torch.stack([b["metrics"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


def build_model(task: str, metric_dim: int, device: torch.device) -> tuple[nn.Module, dict[str, nn.Module | None]]:
    components: dict[str, nn.Module | None] = {}
    if task == "codebert_only":
        components["text"] = build_codebert_encoder(CodeBERTConfig(freeze=True)).to(device)
        model = TextOnlyClassifier().to(device)
    elif task == "gnn_only":
        components["graph"] = build_gnn_encoder(GNNEncoderConfig(in_dim=32)).to(device)
        model = GraphOnlyClassifier().to(device)
    elif task == "multimodal_fusion":
        components["text"] = build_codebert_encoder(CodeBERTConfig(freeze=True)).to(device)
        components["graph"] = build_gnn_encoder(GNNEncoderConfig(in_dim=32)).to(device)
        model = MultimodalClassifier(FusionConfig(metric_dim=metric_dim)).to(device)
    else:
        raise ValueError(f"Unknown neural task: {task}")
    return model, components


def train_neural_model(df: pd.DataFrame, cfg: NeuralTrainConfig, device: torch.device | None = None) -> tuple[nn.Module, dict, list[float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ds = DefectDataset(df, cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate)
    model, components = build_model(cfg.task, metric_dim=len(ds.metric_cols), device=device)

    params = list(model.parameters())
    for comp in components.values():
        if comp is not None and cfg.task.endswith("_finetune"):
            params += [p for p in comp.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    losses: list[float] = []
    model.train()
    for comp in components.values():
        if comp is not None:
            comp.train()

    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            opt.zero_grad()
            y = batch["y"].to(device)
            if cfg.task == "codebert_only":
                text_enc = components["text"]
                assert text_enc is not None
                emb = text_enc(batch["texts"])
                logits = model(emb)
            elif cfg.task == "gnn_only":
                graph_enc = components["graph"]
                assert graph_enc is not None
                embs = []
                for g in batch["graphs"]:
                    nf = g.node_features.to(device)
                    ei = g.edge_index.to(device)
                    embs.append(graph_enc(nf, ei))
                graph_emb = torch.stack(embs)
                logits = model(graph_emb)
            else:
                text_enc = components["text"]
                graph_enc = components["graph"]
                assert text_enc is not None and graph_enc is not None
                text_emb = text_enc(batch["texts"])
                graph_embs = []
                for g in batch["graphs"]:
                    graph_embs.append(graph_enc(g.node_features.to(device), g.edge_index.to(device)))
                graph_emb = torch.stack(graph_embs)
                metric_vec = batch["metrics"].to(device)
                logits = model(text_emb=text_emb, graph_emb=graph_emb, metric_vec=metric_vec)

            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(1, n_batches))

    return model, components, losses


@torch.no_grad()
def predict_neural(model: nn.Module, components: dict, df: pd.DataFrame, cfg: NeuralTrainConfig, device: torch.device | None = None) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DefectDataset(df, cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate)
    model.eval()
    for comp in components.values():
        if comp is not None:
            comp.eval()

    probs: list[float] = []
    for batch in loader:
        if cfg.task == "codebert_only":
            text_enc = components["text"]
            assert text_enc is not None
            emb = text_enc(batch["texts"])
            logits = model(emb)
        elif cfg.task == "gnn_only":
            graph_enc = components["graph"]
            assert graph_enc is not None
            embs = [graph_enc(g.node_features.to(device), g.edge_index.to(device)) for g in batch["graphs"]]
            logits = model(torch.stack(embs))
        else:
            text_enc = components["text"]
            graph_enc = components["graph"]
            assert text_enc is not None and graph_enc is not None
            text_emb = text_enc(batch["texts"])
            graph_embs = [graph_enc(g.node_features.to(device), g.edge_index.to(device)) for g in batch["graphs"]]
            logits = model(
                text_emb=text_emb,
                graph_emb=torch.stack(graph_embs),
                metric_vec=batch["metrics"].to(device),
            )
        p = torch.sigmoid(logits).cpu().numpy().tolist()
        probs.extend(p if isinstance(p, list) else [float(p)])
    return np.array(probs, dtype=float)
