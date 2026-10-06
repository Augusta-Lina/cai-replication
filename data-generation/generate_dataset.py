# Block 1: Imports & Arguments
# -----------------------------

# Stage 1–2 of Constitutional AI: critique–revision generation and dataset construction.
# Rewrite of llm-swarm's examples/constitutional-ai/generate_dataset.py (commit c3249db) for one GPU.
# Differences from the original are listed in SPEC.md.
# Run on the pod, with scripts/serve_model.sh running in another terminal:
#   uv run data-generation/generate_dataset.py --max_samples 2

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import requests
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, HfArgumentParser

api = HfApi()


# Block 1: command-line arguments
@dataclass
class Args:
    max_samples: int = 128
    """Samples to generate per split (-1 for all)"""
    max_new_tokens: int = 1500
    """Max new tokens per generation"""
    temperature: float = 1.0
    """Generation temperature"""
    constitution_path: str = "data-generation/constitution_anthropic.json"
    """Path to the constitution"""
    model_path: str = "/workspace/models/Mistral-7B-Instruct-v0.1"
    """Local model folder (for the tokenizer and its chat template)"""
    server_url: str = "http://localhost:8000/v1/completions"
    """The vLLM server's completions endpoint"""
    output_dir: str = "/workspace/outputs"
    """Where to save the generated datasets as JSONL files"""
    repo_id: str = "cai-conversation-dev"
    """The Hub repo id to push to"""
    timestamp: bool = True
    """Whether to add a timestamp to repo_id"""
    push_to_hub: bool = False
    """Whether to push to the Hub"""


parser = HfArgumentParser((Args,))
(args,) = parser.parse_args_into_dataclasses()
if args.timestamp:
    args.repo_id += str(int(time.time()))

# Block 2: Tokenizer & Cnstitution
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
with open(args.constitution_path) as f:
    data = json.load(f)
    constitutions = data["constitutions"]
    system_chat = data["system_chat"]
    system_chat = [item for sublist in system_chat for item in sublist]

# Block 3: Load Prompts from hh-rlhf
# -----------------------------

# Block 3: prompts (first "Human:" turn of each hh-rlhf harmless-base conversation)
ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
for key in ds:
    max_samples = len(ds[key]) if args.max_samples == -1 else args.max_samples
    ds[key] = ds[key].select(range(max_samples))


def extract(example):
    example = example["chosen"]
    split_text = example.split("\n\n")
    for segment in split_text:
        if "Human:" in segment:
            return {"prompt": segment.split(": ")[1]}


ds = ds.map(extract)
ds = ds.remove_columns(["chosen", "rejected"])
