"""Build and freeze prompt manifests; input IDs can also be supplied offline."""

import hashlib
import json
from itertools import islice
from pathlib import Path


# Workload definition for this survey. Local manifests can reproduce an older suite.
SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test"),
    "mbpp": ("google-research-datasets/mbpp", "sanitized", "test"),
    "lcb": ("livecodebench/code_generation_lite", "release_v5", "test"),
    "aime25": ("yentinglin/aime_2025", "default", "train"),
    "mtbench": ("HuggingFaceH4/mt_bench_prompts", None, "train"),
    "alpaca": ("tatsu-lab/alpaca", None, "train"),
}
LCB_REV = "b6e6b08f8404c4cc696f6251cd3151cda6b43fc7"


def digest(path):
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            sha.update(chunk)
        return sha.hexdigest()


def public_rows(name, limit):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo, config, split = SOURCES[name]
    if name == "lcb":
        # Select only prompt columns; private tests are large and unused.
        rev = LCB_REV
        pattern = f"hf://datasets/{repo}@{rev}/release_v5/test-*.parquet"
        data = load_dataset("parquet", data_files={split: pattern}, split=split,
                            streaming=True, columns=["platform", "question_id", "question_content", "starter_code"])
    else:
        rev = HfApi().dataset_info(repo).sha
        data = load_dataset(repo, config, split=split, revision=rev)
    rows = []
    for index, row in enumerate(islice(data, limit)):
        key = next((row[k] for k in ("task_id", "unique_id", "prompt_id", "id") if k in row), index)
        if name in {"gsm8k", "math500", "aime25"}:
            key = index if name == "aime25" else key
            field = "question" if name == "gsm8k" else "problem"
            text = row[field] + "\nShow your reasoning and write the final answer in \\boxed{}."
        elif name == "humaneval":
            text = "Complete the following Python function.\n```python\n" + row["prompt"] + "\n```"
        elif name == "mbpp":
            text = row["prompt"]
        elif name == "lcb":
            key = f"{row['platform']}/{row['question_id']}"
            text = row["question_content"] + "\nWrite a Python solution."
            if row["starter_code"]:
                text += "\n```python\n" + row["starter_code"] + "\n```"
        elif name == "mtbench":
            text = row["prompt"][0]  # fixed first-turn throughput workload
        else:
            text = row["instruction"]
            if row["input"].strip():
                text += "\n\n" + row["input"]
        rows.append({"id": str(key), "prompt": text})
    return rows, {"dataset": repo, "revision": rev, "config": config, "split": split, "selection": "first_n"}


def local_rows(directory, name, limit):
    path = Path(directory) / (name + ".jsonl")
    with path.open(encoding="utf-8") as stream:
        rows = list(islice((json.loads(line) for line in stream if line.strip()), limit))
    return rows, {"file": str(path.resolve()), "sha256": digest(path), "selection": "first_n"}


def render(row, tokenizer, thinking=False):
    if isinstance(row, str):
        row = {"prompt": row}
    if not isinstance(row, dict):
        raise ValueError("Each JSONL line must be a prompt string or a JSON object")
    if "input_ids" in row or "prompt_token_ids" in row:
        ids = row.get("input_ids", row.get("prompt_token_ids"))
    elif "formatted_prompt" in row:
        ids = tokenizer.encode(row["formatted_prompt"], add_special_tokens=False)
    else:
        messages = row.get("messages")
        if messages is None:
            text = next((row[key] for key in ("prompt", "turns", "question", "instruction", "text")
                         if key in row), None)
            if isinstance(text, list) and text:
                text = text[0]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Provide input_ids, formatted_prompt, messages or prompt in each record")
            if "instruction" in row and row.get("input"):
                text += "\n\n" + row["input"]
            messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=thinking)
        ids = tokenizer.encode(formatted, add_special_tokens=False)
    if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("Expected a nonempty one-dimensional list of nonnegative token IDs")
    return ids
