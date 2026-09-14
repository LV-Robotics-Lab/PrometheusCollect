#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UPLOAD_PYTHON="${UPLOAD_PYTHON:-${PROJECT_ROOT}/.venv-modelscope-upload/bin/python}"

if [[ ! -x "${UPLOAD_PYTHON}" ]]; then
  echo "Upload environment missing. Run from the project directory:" >&2
  echo "uv venv --python /usr/bin/python3 .venv-modelscope-upload" >&2
  echo "uv pip install --python .venv-modelscope-upload/bin/python -r scripts/requirements-upload.txt" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
exec "${UPLOAD_PYTHON}" -m prometheus.workflows.upload_dataset "$@"
