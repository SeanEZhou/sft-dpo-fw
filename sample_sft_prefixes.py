import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer

from logger import logger
from utils import read_json_file, read_yaml_config

# === Config ===
DATASET_NAME = "HuggingFaceFW/fineweb"
SPLIT = "train"
NUM_PREFIXES = 5000
OUTPUT_FILE = "prefixes_fineweb.jsonl"
PREFIX_TOKENS = 64
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
COLUMN_NAME = "text"


def sample_sft_prefixes(
    dataset_path: str,
    split: str,
    column_name: str,
    num_prefixes: int,
    output_file: str,
    prefix_tokens: int,
    model_name: str,
):
    if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
        # === Load dataset from local JSON/JSONL file ===
        dataset = read_json_file(dataset_path)
    else:
        # === Load dataset in streaming mode (no full download) ===
        dataset = load_dataset(dataset_path, split=split, streaming=True)

    # === Load tokenizer ===
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # === Sample prefixes ===
    prefixes = []
    for example in dataset:
        if len(prefixes) >= num_prefixes:
            break

        text = example[column_name].strip()
        if not text:
            continue

        tokens = tok.encode(text)
        if len(tokens) < prefix_tokens:
            continue

        truncated = tokens[:prefix_tokens]
        decoded = tok.decode(truncated, skip_special_tokens=True)
        prefixes.append(decoded)

    # === Save ===
    with open(output_file, "w", encoding="utf-8") as f:
        for p in prefixes:
            f.write(json.dumps({"prefix": p}) + "\n")

    logger.info(
        "Saved prefixes.",
        prefixes_file=output_file,
        count=len(prefixes),
        example_prefix=prefixes[0],
    )


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation with local models"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="qwen1b/config_sample_sft_prefixes.yaml",
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()
    config = read_yaml_config(args.config_path)

    dataset_path = config.get("dataset_path", DATASET_NAME)
    split = config.get("split", SPLIT)
    num_prefixes = config.get("num_prefixes", NUM_PREFIXES)
    output_file = config.get("output_file", OUTPUT_FILE)
    prefix_tokens = config.get("prefix_tokens", PREFIX_TOKENS)
    model_name = config.get("model_name", MODEL_NAME)
    column_name = config.get("column_name", COLUMN_NAME)

    sample_sft_prefixes(
        dataset_path=dataset_path,
        split=split,
        column_name=column_name,
        num_prefixes=num_prefixes,
        output_file=output_file,
        prefix_tokens=prefix_tokens,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
