# Part 1: header and the generation stage
# Spec: original component → replacement

What changed between Hugging Face's Constitutional AI pipeline and this replication, and why.
Everything not listed here is kept as in the original.

Original sources (pinned):
- `llm-swarm` @ `c3249db`: `examples/constitutional-ai/generate_dataset.py`, `constitution_anthropic.json`
- `alignment-handbook` @ `53c11c3`: `scripts/sft.py`, `scripts/dpo.py`, `recipes/constitutional-ai/`

## Stage 1–2: data generation (`generate_dataset.py`)

| Original | Problem for us | Replacement |
|---|---|---|
| `LLMSwarm(isc)` starts TGI servers on a Slurm cluster (6–64 GPUs) | no cluster | one RunPod A40 pod; `scripts/serve_model.sh` starts a vLLM server on it |
| Model fetched by TGI at startup | — | downloaded once onto the persistent volume by `scripts/download_model.py` (`/workspace/models/Mistral-7B-Instruct-v0.1`) |
| `AsyncInferenceClient(model=llm_swarm.endpoint)` + `client.text_generation(prompt, max_new_tokens, stop_sequences, temperature)` (lines 67, 83–88) | TGI-specific client | `POST http://localhost:8000/v1/completions` with `{"model": "mistral", "prompt", "max_tokens", "stop", "temperature"}`; answer in `choices[0].text`, token count in `usage.completion_tokens` |
| `async def process_text`, `asyncio.gather`, `Semaphore` (lines 66, 70, 101–105) | throughput machinery for 45k prompts | plain `def`; `ThreadPoolExecutor` with `--max_workers` (default 8) plays the Semaphore's role; results sorted by `(split, i)`; failed requests retried twice. `--max_workers 1` gives the sequential baseline. Measured: 54 tok/s at 1 worker, 209 tok/s at 8 |
| `HfArgumentParser((Args, LLMSwarmConfig))` (line 36) | `LLMSwarmConfig` gone with llm-swarm | `HfArgumentParser((Args,))` |
| `rate_limit = 500 * isc.instances` (line 64) | unused even in the original | removed |
| `AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")` (line 40) | fetches from the Hub each run | load from the local model folder (same files) |
| `max_samples` default 128 (line 20) | — | kept; first test runs use `--max_samples 2` |
| `constitution_anthropic.json` | — | pasted unchanged into `data-generation/`; source is the pinned llm-swarm commit above. Own constitutions are added as separate files and selected with `--constitution_path` |
| `token_length` counted with `tokenizer.encode` (line 94) | — | taken from the server's `usage.completion_tokens` |
| Bug: `ds.remove_columns([...])` result discarded (line 63) | no-op | `ds = ds.remove_columns([...])` |
| Bug: `repo_id` undefined when `--repo_id` contains `/` (lines 142–148) | `NameError` | always set `repo_id` before use |
| Output: `prompt`, `messages`, `chosen`, `rejected`; halves pushed as `{split}_sft` / `{split}_prefs` (lines 116–141) | — | **kept identical** (the contract with the training configs) |
| `push_to_hub` + upload of script and constitution (lines 139–150) | needs a Hub login | kept; HF Write token in `.env` (`HF_TOKEN`), created when first needed |
| Library versions from early 2024 | we have `transformers` 5.x, `datasets` 5.x, `huggingface_hub` 1.x | check each call still exists while writing |

# Part 2: the training stage

## Stage 3–4: training (`sft.py`, `dpo.py`, the two YAMLs)

Not started. Known changes so far:

| Original | Problem for us | Replacement |
|---|---|---|
| Full fine-tune of Mistral-7B on 8×H100 (DeepSpeed ZeRO-3) | budget | LoRA on one A40, switched on by YAML keys (`use_peft`, `lora_r`, …) |
| `model_name_or_path: alignment-handbook/mistral-7b-sft-constitutional-ai` in the DPO yaml | HF's model | our own SFT output |
| `dataset_mixture` ids `HuggingFaceH4/cai-conversation-harmless` | HF's dataset | our dataset id; split and column names unchanged |
| UltraChat / UltraFeedback mixtures (~23k / ~63k rows) | drown a 128-sample dataset | drop or shrink |
| `test_split_size` 1000 / 3000 | larger than our dataset | small number or `null` |
| `eval_steps`, `save_steps`, learning rate, epochs | tuned for full scale | retune for tiny data |
| `dataset_num_proc: 12` | CPU count | lower |
| `attn_implementation: flash_attention_2`, `bf16` | CUDA-only, fine on the A40 | kept if the pod's PyTorch supports them |
| Two full models in DPO (`model` + `ref_model`) | double memory | LoRA with `ref_model=None` (needs a `dpo.py` edit) |

# Part 3: infrastructure notes

## Infrastructure

| Topic | Decision |
|---|---|
| Pod | RunPod, A40 (48 GB), on-demand, template Runpod Pytorch 2.8.0, 30 GB container disk |
| Storage | Global volume `cai_workspace` at `/workspace` for big artifacts (model, outputs). Code and `.venv` live on the container disk, rebuilt from GitHub each session |
| Volume quirks | no file-permission changes allowed (harmless warnings from downloaders; uv needs `UV_CACHE_DIR` on the container disk). Slow for large reads: vLLM took 16 min to load 14 GB. Planned fix: copy the model to the container disk at session start |
| Pod bootstrap | `git clone` → `source scripts/setup_pod.sh` (uv, cache dir, `uv sync`, `pip install vllm`) |
| Model server | vLLM, OpenAI-compatible API on port 8000. Installed on the pod only (Linux/CUDA), so not in `pyproject.toml` |
| Git | the pod only pulls; commits and pushes happen on the iMac |
