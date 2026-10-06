# Pod setup: run once per pod session, after cloning the repo.
# Installs uv and vLLM, keeps caches off the Global volume (it refuses chmod), and rebuilds the Python environment.
# Usage (from the repo folder):   source scripts/setup_pod.sh

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# The template points these caches at /workspace (Global volume), which breaks permission changes.
export UV_CACHE_DIR=/root/.cache/uv
export HF_HOME=/root/.cache/huggingface

uv sync

# vLLM goes on the container disk (not in pyproject: Linux/CUDA only, and 3 GB)
pip install vllm

echo "Pod setup complete. Run scripts with: uv run data-generation/<name>.py"
