#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    cat <<'EOF'
Error: uv is not installed.

Install instructions: https://github.com/astral-sh/uv#installation
Once uv is installed, re-run this script.
EOF
    exit 1
fi

VENV_DIR="${SCRIPT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    uv venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

INSTALL_TARGET="talkito"
if [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
    INSTALL_TARGET="${SCRIPT_DIR}"
fi

echo "Installing ${INSTALL_TARGET} with uv..."
uv pip install --upgrade "${INSTALL_TARGET}"

cat <<'EOF'
Installation complete!

To use this environment, run:
  source .venv/bin/activate

Then invoke TalkiTo commands normally, e.g.:
  talkito --help
EOF
