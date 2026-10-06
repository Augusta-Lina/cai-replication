# Start the vLLM model server for Mistral-7B-Instruct-v0.1 on port 8000.
# Replaces llm-swarm + TGI from the original pipeline. Loading from /workspace takes ~15 min.
# Usage (from the repo folder):   bash scripts/serve_model.sh
# Leave it running in its own terminal; send requests to http://localhost:8000/v1/completions

vllm serve /workspace/models/Mistral-7B-Instruct-v0.1 \
  --served-model-name mistral \
  --port 8000
