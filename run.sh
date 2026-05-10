#!/usr/bin/env bash
# Launches the Reel Transcriber UI with macOS fork-safety workarounds
# (Python 3.9 + multi-threaded native libs like mlx-whisper crash on fork
# without these env vars. See: macOS objc fork safety.)

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

exec streamlit run app.py \
  --server.port 8501 \
  --browser.gatherUsageStats false
