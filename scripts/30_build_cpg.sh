#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-}"
OUT_DIR="${2:-data/processed/cpg}"

if [[ -z "${INPUT_DIR}" ]]; then
  echo "Usage: scripts/30_build_cpg.sh <input_source_dir> [output_dir]"
  exit 1
fi

mkdir -p "${OUT_DIR}"
python -c "
from pathlib import Path
from sdp.cpg.joern_export import export_cpg_with_joern
import sys
input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
records = export_cpg_with_joern(input_dir, output_dir)
print(f'CPG export records: {len(records)}')
for r in records[:5]:
    print(r)
" "${INPUT_DIR}" "${OUT_DIR}"
