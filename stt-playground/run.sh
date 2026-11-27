#!/usr/bin/env bash
# Start the playground. Override the checkpoint: STT_CHECKPOINT=/path/to/ckpt ./run.sh
cd "$(dirname "$0")" || exit 1
# CTranslate2 needs the CUDA 12 cuBLAS/cuDNN that the torch wheels bring into the venv.
LIBS=$(find .venv/lib -type d -path '*site-packages/nvidia/*/lib' 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:-}"
exec uv run python stt_ui.py
