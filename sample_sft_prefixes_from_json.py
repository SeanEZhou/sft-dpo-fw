import argparse
import json

from transformers import AutoTokenizer

from logger import logger
from utils import read_json_file, read_yaml_config


def sample_sft_prefixes(
    dataset_path: str,
    column_name: str,
    num_prefixes: int,
    output_file: str,
    prefix_tokens: int,
    model_name: str,
):
    """Sample prefixes from a dataset for SFT training."""

    logger.info(
        "Sampling prefixes for SFT training...",
        dataset_path=dataset_path,
        num_prefixes=num_prefixes,
        output_file=output_file,
        prefix_tokens=prefix_tokens,
        model_name=model_name,
    )

    # === Load dataset in streaming mode (no full download) ===
    dataset_generator = read_json_file(dataset_path)

    # === Load tokenizer ===
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # === Sample prefixes ===
    prefixes = []
    for example in dataset_generator:
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
        "Finished sampling prefixes.",
        output_file=output_file,
        num_prefixes=len(prefixes),
        example_prefix=prefixes[0] if prefixes else None,
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

    dataset_path = config.get("dataset_path")
    num_prefixes = config.get("num_prefixes")
    output_file = config.get("output_file")
    prefix_tokens = config.get("prefix_tokens")
    model_name = config.get("model_name")
    column_name = config.get("column_name")

    logger.info("Starting sampling with configuration:", config=config)

    sample_sft_prefixes(
        dataset_path=dataset_path,
        num_prefixes=num_prefixes,
        output_file=output_file,
        prefix_tokens=prefix_tokens,
        model_name=model_name,
        column_name=column_name,
    )


if __name__ == "__main__":
    main()
