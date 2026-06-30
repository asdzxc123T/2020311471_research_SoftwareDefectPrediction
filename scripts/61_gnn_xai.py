from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.graph.builder import build_code_graph
from sdp.models.gnn.gat_encoder import GNNEncoderConfig, build_gnn_encoder
from sdp.models.fusion.multimodal import GraphOnlyClassifier
from sdp.utils.paths import ensure_dirs, find_repo_root
from sdp.xai.gnn_explainer import explain_graph


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/gnn_only.yaml")
    ap.add_argument("--row_idx", type=int, default=0)
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    eval_cfg = cfg.get("eval", {}) or {}
    ds_path = Path(eval_cfg.get("dataset_path", "data/processed/sample_gnn.parquet"))
    df = read_table(root / ds_path)
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No test rows")

    row = test_df.iloc[args.row_idx]
    code = str(row.get("code_text") or row.get("diff_text") or "pass")
    language = str(row.get("language") or "") if "language" in row.index else None
    graph = build_code_graph(code, language=language or None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_enc = build_gnn_encoder(GNNEncoderConfig(in_dim=32)).to(device)
    clf = GraphOnlyClassifier().to(device)

    model_path = root / eval_cfg.get("model_path", "artifacts/models/gnn_only.pt")
    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        graph_enc.load_state_dict(ckpt["components"]["graph"])
        clf.load_state_dict(ckpt["model_state"])

    nf = graph.node_features.to(device)
    ei = graph.edge_index.to(device)
    exp = explain_graph(clf, graph_enc, nf, ei)

    out_dir = root / "reports" / "xai"
    ensure_dirs(out_dir)
    out_json = out_dir / "gnn_explainer.json"
    out_json.write_text(
        json.dumps(
            {
                "row_idx": args.row_idx,
                "function_id": row.get("function_id"),
                "node_importance": exp.node_importance.tolist(),
                "edge_importance": exp.edge_importance.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(6, 3))
    plt.bar(range(len(exp.node_importance)), exp.node_importance.numpy())
    plt.title("GNNExplainer node importance")
    plt.tight_layout()
    plt.savefig(out_dir / "gnn_explainer_nodes.png", dpi=200)
    plt.close()
    print(f"Wrote: {out_json.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
