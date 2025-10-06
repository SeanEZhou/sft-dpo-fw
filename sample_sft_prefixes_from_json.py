import argparse
import json
from typing import Dict, List

import yaml
from transformers import AutoTokenizer

from utils import read_json_file, read_yaml_config


def sample_sft_prefixes(
    dataset_path: str,
    num_prefixes: int,
    output_file: str,
    prefix_tokens: int,
    model_name: str,
):
    """Sample prefixes from a dataset for SFT training."""
    DATASET_PATH = dataset_path
    NUM_PREFIXES = num_prefixes
    OUTPUT_FILE = output_file
    PREFIX_TOKENS = prefix_tokens
    MODEL_NAME = model_name

    print(f"Dataset path: {DATASET_PATH}")
    print(f"Number of prefixes to sample: {NUM_PREFIXES}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Prefix tokens: {PREFIX_TOKENS}")
    print(f"Model name: {MODEL_NAME}")

    # === Load dataset in streaming mode (no full download) ===
    dataset_generator = read_json_file(DATASET_PATH)

    # === Load tokenizer ===
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # === Sample prefixes ===
    prefixes = []
    for example in dataset_generator:
        if len(prefixes) >= NUM_PREFIXES:
            break
        # (Pdb) example.keys()
        # dict_keys(['prompt_id', 'file_path', 'pattern_desc', 'cwe_identifier', 'rule', 'analyzer', 'pattern_id', 'line_number', 'line_text', 'test_case_prompt', 'origin_code', 'language', 'variant', 'repo'])
        text = example["test_case_prompt"].strip()
        if not text:
            continue

        tokens = tok.encode(text)
        if len(tokens) < PREFIX_TOKENS:
            continue

        truncated = tokens[:PREFIX_TOKENS]
        decoded = tok.decode(truncated, skip_special_tokens=True)
        prefixes.append(decoded)

    # === Save ===
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in prefixes:
            f.write(json.dumps({"prefix": p}) + "\n")

    print(f"Saved {len(prefixes)} prefixes to {OUTPUT_FILE}")
    print("Example prefix:\n", prefixes[0])


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

    dataset_path = config.get("dataset_path")
    num_prefixes = config.get("num_prefixes")
    output_file = config.get("output_file")
    prefix_tokens = config.get("prefix_tokens")
    model_name = config.get("model_name")

    print(f"Configuration: {config}")

    sample_sft_prefixes(
        dataset_path=dataset_path,
        num_prefixes=num_prefixes,
        output_file=output_file,
        prefix_tokens=prefix_tokens,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
