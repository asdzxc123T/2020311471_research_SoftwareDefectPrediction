#!/usr/bin/env bash
set -euo pipefail

if ! command -v joern >/dev/null 2>&1; then
  echo "joern not found in PATH."
  echo "Start the optional joern container or install Joern and retry."
  exit 1
fi

INPUT_DIR="${1:-}"
OUT_DIR="${2:-data/processed/cpg}"

if [[ -z "${INPUT_DIR}" ]]; then
  echo "Usage: scripts/30_build_cpg.sh <input_source_dir> [output_dir]"
  exit 1
fi

mkdir -p "${OUT_DIR}"
echo "This script is a minimal entrypoint for Joern-based CPG generation."
echo "Input: ${INPUT_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "Implement concrete joern-import / export commands per target language/repo."

