#!/bin/sh
# Install CPython 3.12 and the pinned torch wheel into ~/ppdl-w-bench/venv.
set -eu
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
ROOT="$HOME/ppdl-w-bench"
mkdir -p "$ROOT"
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        PATH="$HOME/.local/bin:$PATH"
    elif [ -x "$HOME/.local/share/mise/shims/uv" ]; then
        PATH="$HOME/.local/share/mise/shims:$PATH"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        PATH="$HOME/.local/bin:$PATH"
    fi
fi
uv python install 3.12
if [ ! -x "$ROOT/venv/bin/python" ]; then
    uv venv --python 3.12 "$ROOT/venv"
fi
PY="$ROOT/venv/bin/python"
if [ "$(uname -s)" = "Darwin" ]; then
    uv pip install --python "$PY" 'torch==2.14.1'
else
    uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu126 'torch==2.14.1'
fi
"$PY" -c 'import torch; print(torch.__version__); print("cuda", torch.version.cuda)'
