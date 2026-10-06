# Constitutional AI replication

A small-scale replication of Hugging Face's Constitutional AI pipeline
(https://huggingface.co/blog/constitutional_ai), built to run on a single rented GPU e.g. RunPod instead of a cluster, while staying compatible with the original scripts.

The replication was made on RunPod, using a single A40 GPU with a persistent Global volume mounted at `/workspace`.

## Folder Structure

```
cai-replication/
├── README.md             ← how to use the repo: "clone, uv sync, run scripts/…"
├── scripts/              ← Setup and one-off tasks
│   └── download_model.py ← Download Mistral
│   └── setup_pod.sh      ← Pod setup (uv, cache, uv sync)
└── data-generation/      ← Data generation pipeline, Stages 1–2
    └── generate_dataset.py (planned)
```

## 1. Setup

Requires [uv](https://docs.astral.sh/uv/). On your own machine, after cloning: `uv sync`.

On a RunPod pod (fresh container each session):

```bash
cd /root
git clone https://github.com/Augusta-Lina/cai-replication.git
cd cai-replication
source scripts/setup_pod.sh
```

## 2. Scripts

### 2.1 Download the model

Downloads Mistral-7B-Instruct-v0.1 (about 14 GB) to `/workspace/models/`. Run on the GPU pod,
where `/workspace` is the persistent volume:

```bash
uv run scripts/download_model.py
```

### 2.2. Pod Setup

Installs uv, points its cache at the container disk (the `/workspace` volume does not allow setting file permissions), and runs `uv sync`. Must be run with `source`, not `bash`, so the environment settings persist in your terminal.

```bash
source scripts/setup_pod.sh
```
