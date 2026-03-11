#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${ROOT_DIR}/data/raw/defects4j"

mkdir -p "${DEST_DIR}"

if [[ -d "${DEST_DIR}/defects4j/.git" ]]; then
  echo "Defects4J repo already exists at: ${DEST_DIR}/defects4j"
  exit 0
fi

echo "Cloning Defects4J into: ${DEST_DIR}/defects4j"
git clone --depth 1 https://github.com/rjust/defects4j.git "${DEST_DIR}/defects4j"

echo "NOTE: Defects4J requires additional dependencies (perl, Java, build tools)."
echo "See: https://github.com/rjust/defects4j"

