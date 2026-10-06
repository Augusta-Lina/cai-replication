# Constitutional AI replication

A small-scale replication of Hugging Face's Constitutional AI pipeline
(https://huggingface.co/blog/constitutional_ai), built to run on a single rented GPU e.g. RunPod instead of a cluster, while staying compatible with the original scripts.

The replication was made using RodPod, a single A40 GPU using a persistent volume of 30GB

## 1. Setup

Requires [uv](https://docs.astral.sh/uv/). After cloning:

```bash
uv sync
```

This installs the pinned Python version and all packages from `pyproject.toml` and `uv.lock`.

## 2. Scripts

### 2.1 Download the model

Downloads Mistral-7B-Instruct-v0.1 (about 14 GB) to `/workspace/models/`. Run on the GPU pod,
where `/workspace` is the persistent volume:

```bash
uv run scripts/download_model.py
```
