# Download Mistral-7B-Instruct-v0.1 onto the pod's persistent volume.
# Fetches the safetensors weights, config and tokenizer files (about 14 GB),
# skipping the duplicate .bin weights. Run on the pod with: uv run scripts/download_model.py

# Block 1: Download Mistral Instruct from HuggingFace & Save to Pod's Workspace
from huggingface_hub import snapshot_download

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
LOCAL_DIR = "/workspace/models/Mistral-7B-Instruct-v0.1"

path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
    ignore_patterns=["pytorch_model*"],
)

print(f"Model downloaded to: {path}")