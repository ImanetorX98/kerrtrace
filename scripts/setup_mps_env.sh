#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup_mps_env.sh [python_bin] [venv_dir]
# Example:
#   ./scripts/setup_mps_env.sh /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 .venv-mps

PY_BIN="${1:-python3.11}"
VENV_DIR="${2:-.venv-mps}"

if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  echo "Python binary not found: ${PY_BIN}"
  exit 1
fi

echo "Creating venv: ${VENV_DIR} (python: ${PY_BIN})"
"${PY_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip
python -m pip install -r requirements.txt

echo
echo "Checking torch backends..."
python - <<'PY'
import platform
import torch
print("Platform:", platform.platform())
print("Machine:", platform.machine())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY

echo
echo "If MPS available is True, run renders with: --device mps"
