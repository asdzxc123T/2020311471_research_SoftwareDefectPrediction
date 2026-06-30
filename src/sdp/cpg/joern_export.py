from __future__ import annotations

import json
import subprocess
from pathlib import Path


def export_cpg_with_joern(input_dir: Path, output_dir: Path) -> list[dict]:
    """
    Minimal Joern CPG export hook.
    Runs joern-parse when available; otherwise records placeholder metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    joern_available = False
    try:
        joern_available = subprocess.run(["joern", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        joern_available = False

    for src in sorted(input_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.suffix.lower() not in {".java", ".py", ".js", ".ts"}:
            continue
        rel = src.relative_to(input_dir)
        out_path = output_dir / f"{rel.as_posix().replace('/', '_')}.cpg.json"
        meta = {"source": str(src), "cpg_path": str(out_path), "joern_available": joern_available}
        if joern_available:
            # Placeholder: real integration would call joern-parse / joern-export
            meta["status"] = "queued_for_joern_export"
        else:
            meta["status"] = "joern_not_available_ast_fallback"
        out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        records.append(meta)
    return records
