# Pod setup: run once per pod session, after cloning the repo.
# Installs uv, keeps its cache off the network volume, and rebuilds the Python environment.
# Usage (from the repo folder):   source scripts/setup_pod.sh

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

export UV_CACHE_DIR=/root/.cache/uv
uv sync

echo "Pod setup complete. Run scripts with: uv run scripts/<name>.py"
