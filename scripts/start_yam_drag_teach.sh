#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Prefer the repository environment so this helper works without manually
# activating it. An already selected PYTHON_BIN still takes precedence.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_EXECUTABLE="python"
fi

exec "${PYTHON_EXECUTABLE}" "${SCRIPT_DIR}/yam_drag_teach.py" "$@"
