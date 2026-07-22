#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Make matplotlib cache writable + deterministic
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/run_all.py --stage all

echo "OK: outputs in results/ and plots/publication/"
