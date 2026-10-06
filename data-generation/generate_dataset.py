# Block 1: Imports & Arguments
# -----------------------------

# Stage 1–2 of Constitutional AI: critique–revision generation and dataset construction.
# Rewrite of llm-swarm's examples/constitutional-ai/generate_dataset.py (commit c3249db) for one GPU.
# Differences from the original are listed in SPEC.md.
# Run on the pod, with scripts/serve_model.sh running in another terminal:
#   uv run data-generation/generate_dataset.py --max_samples 2

import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import requests
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, HfArgumentParser

api = HfApi()
load_dotenv()

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

# first "Human:" turn of each hh-rlhf harmless-base conversation
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

# Block 4: one call to the vLLM server (replaces client.text_generation)
STOP_SEQ = ["User:", "###", "<|endoftext|>"]


def generate(chat):
    """Send the chat so far to the server; return the completion and its token count."""
    response = requests.post(
        args.server_url,
        json={
            "model": "mistral",
            "prompt": tokenizer.apply_chat_template(chat, tokenize=False),
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "stop": STOP_SEQ,
            "add_special_tokens": False,
        },
        timeout=600,
    )
    response.raise_for_status()
    result = response.json()
    completion = result["choices"][0]["text"]
    for stop_seq in STOP_SEQ:
        if completion.endswith(stop_seq):
            completion = completion[: -len(stop_seq)].rstrip()
    return completion, result["usage"]["completion_tokens"]


# Block 5: Critique-revision for One Prompt
# -----------------------------

# Run a 3 part loop once to generate a critique and revision for one prompt
def process_text(split, i, task):
    chat = system_chat.copy()
    constitution = random.choice(constitutions)
    token_length = 0
    row = {}
    for prompt, prompt_key, response_key in [
        (task, "init_prompt", "init_response"),
        (constitution["critic"], "critic_prompt", "critic_response"),
        (constitution["revision"], "revision_prompt", "revision_response"),
    ]:
        chat.append({"role": "user", "content": prompt})
        completion, n_tokens = generate(chat)
        chat.append({"role": "assistant", "content": completion})
        token_length += n_tokens
        row[prompt_key] = prompt
        row[response_key] = completion
    return split, i, token_length, row


# Block 6: Run Everything & Build the Dataset
# -----------------------------

# run every prompt, then build the datasets (Stage 2)
def main():
    load_dotenv()
    start_time = time.time()
    results = []
    for split in ds:
        for idx, row in enumerate(ds[split]):
            results.append(process_text(split, idx, row["prompt"]))
            print(f"{split}: {idx + 1}/{len(ds[split])} done", flush=True)
    total_duration = time.time() - start_time
    total_tokens = sum(result[2] for result in results)
    overall_tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
    print(f"Overall Tokens per Second: {overall_tokens_per_second}")

    all_ds = defaultdict(lambda: defaultdict(list))
    for result in results:
        for key, value in result[3].items():
            all_ds[result[0]][key].append(value)

    def process(example):  # where Stage 2 happens
        return {
            "prompt": example["init_prompt"].strip(),
            "messages": [
                {"role": "user", "content": example["init_prompt"].strip()},
                {"role": "assistant", "content": example["revision_response"].strip()},
            ],
            "chosen": [
                {"role": "user", "content": example["init_prompt"].strip()},
                {"role": "assistant", "content": example["revision_response"].strip()},
            ],
            "rejected": [
                {"role": "user", "content": example["init_prompt"].strip()},
                {"role": "assistant", "content": example["init_response"].strip()},
            ],
        }

    os.makedirs(args.output_dir, exist_ok=True)
    for split in all_ds:
        df = pd.DataFrame(all_ds[split])
        print("=" * 10 + split + "=" * 10)
        print(df)
        post_ds = Dataset.from_dict(all_ds[split])
        post_ds = post_ds.map(process)
        post_ds.to_json(f"{args.output_dir}/{split}.jsonl")
        if args.push_to_hub:
            repo_id = args.repo_id
            if "/" not in repo_id:  # find the current user
                repo_id = f"{api.whoami()['name']}/{repo_id}"
            post_ds.select(range(len(post_ds) // 2)).push_to_hub(repo_id, split=f"{split}_sft")
            post_ds.select(range(len(post_ds) // 2, len(post_ds))).push_to_hub(repo_id, split=f"{split}_prefs")
            for file, name in zip([__file__, args.constitution_path], ["create_dataset.py", "constitution.json"]):
                api.upload_file(path_or_fileobj=file, path_in_repo=name, repo_id=repo_id, repo_type="dataset")


main()
