# Render a generated JSONL file (from generate_dataset.py) as a readable Markdown report.
# One section per prompt: prompt, first answer, critique principle, critique, revision.
# Run on the pod, for example:
#   uv run data-generation/render_outputs.py --input /workspace/outputs/train.jsonl
#   uv run data-generation/render_outputs.py --input /workspace/outputs/train.jsonl --rows 0 3 7

import json
from dataclasses import dataclass, field

from transformers import HfArgumentParser


# Block 1: command-line arguments
# -----------------------------
@dataclass
class Args:
    input: str = "/workspace/outputs/train.jsonl"
    """JSONL file written by generate_dataset.py"""
    output: str = ""
    """Markdown file to write (default: same name as input, .md)"""
    rows: list[int] = field(default_factory=list)
    """Row numbers to include (default: all)"""
    max_chars: int = 0
    """Truncate each text to this many characters (0 = no truncation)"""


parser = HfArgumentParser((Args,))
(args,) = parser.parse_args_into_dataclasses()
if not args.output:
    args.output = args.input.rsplit(".", 1)[0] + ".md"


# Block 2: read the rows, keep the selected ones
# -----------------------------

with open(args.input) as f:
    all_rows = [json.loads(line) for line in f]
if args.rows:
    rows = [(i, all_rows[i]) for i in args.rows]
else:
    rows = list(enumerate(all_rows))


def clip(text):
    text = text.strip()
    if args.max_chars and len(text) > args.max_chars:
        return text[: args.max_chars] + " […]"
    return text


# Block 3: write one Markdown section per row
# -----------------------------
split = args.input.rsplit("/", 1)[-1].rsplit(".", 1)[0]
lines = [f"# Generated samples: {split}", "", f"Source: `{args.input}`, {len(rows)} of {len(all_rows)} rows.", ""]
for i, row in rows:
    lines += [f"## Row {i}: {clip(row['prompt'])}", ""]
    lines += ["**Initial answer**", "", clip(row["init_response"]), ""]
    lines += ["**Critique request (the constitution principle drawn)**", "", clip(row["critic_prompt"]), ""]
    lines += ["**Critique**", "", clip(row["critic_response"]), ""]
    lines += ["**Revision request**", "", clip(row["revision_prompt"]), ""]
    lines += ["**Revised answer**", "", clip(row["revision_response"]), ""]
    lines += ["---", ""]

with open(args.output, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {len(rows)} rows to {args.output}")
