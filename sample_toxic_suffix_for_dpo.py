import argparse
import json
import random

from datasets import load_dataset
from transformers import AutoTokenizer

from logger import logger
from utils import get_nested, read_json_file, read_yaml_config

# === Config ===
# DATASET also allowds for loading a local JSON file.
DATASET = "allenai/real-toxicity-prompts"
OUTPUT_FILE = "toxic_suffixes_l8b.jsonl"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
NUM_SUFFIXES = 2000
SEED = 42
SAVE_GOLDEN_PREFIX = True
# Toxicity threshold for filtering suffixes.
TOXICITY_THRESHOLD = 0.7
# Cap suffix length.
MAX_TOKENS = 64
# Optional floor on suffix length to avoid very short suffixes.
MIN_TOKENS = 15
# Keys to access the suffix text in the dataset.
TEXT_KEYS = ["continuation", "text"]
# Keys to access the toxicity score in the dataset.
TOXICITY_KEYS = ["continuation", "toxicity"]
# Keys to access the golden prefix in the dataset.
GOLDEN_PREFIX_KEYS = ["prompt", "text"]


def process_dataset(
    dataset,
    model_name,
    output_file,
    num_suffixes,
    save_golden_prefix,
    toxicity_threshold,
    max_tokens,
    min_tokens,
    text_keys,
    toxicity_keys,
    golden_prefix_keys,
):
    """Process the dataset to extract toxic suffixes based on the given configuration."""

    tok = AutoTokenizer.from_pretrained(model_name)

    # === Load dataset ===
    logger.info("Loading dataset.", dataset=dataset)
    if dataset.endswith(".json") or dataset.endswith(".jsonl"):
        # === Load dataset from local JSON/JSONL file ===
        ds = read_json_file(dataset)
    else:
        # === Load dataset in streaming mode (no full download) ===
        ds = load_dataset(dataset, split="train", streaming=True)

    # === Collect suffixes (continuations) ===
    suffixes = []
    for ex in ds:
        suffix_text = get_nested(ex, text_keys).strip()

        if toxicity_keys is not None:
            tox_score = 1.0  # Default to 1.0 if no toxicity keys are provided.
        else:
            # Some entries don't have a toxicity score. The default is 0.0.
            tox_score = get_nested(ex, toxicity_keys) or 0.0

        if not suffix_text or tox_score < toxicity_threshold:
            continue

        # Tokenizes to control length.
        tokens = tok(suffix_text, truncation=True, max_length=max_tokens)["input_ids"]
        if len(tokens) < min_tokens:
            continue

        # Decodes back to enforce truncation at token level.
        suffix_trimmed = tok.decode(tokens, skip_special_tokens=True).strip()

        entry = {"suffix": suffix_trimmed}
        if save_golden_prefix:
            entry["golden_prefix"] = get_nested(ex, golden_prefix_keys).strip()

        suffixes.append(entry)
        if len(suffixes) >= num_suffixes:
            break

    logger.info("Collected candidate suffixes.", count=len(suffixes))

    # === Save ===
    with open(output_file, "w") as f:
        for s in suffixes:
            f.write(json.dumps(s) + "\n")

    logger.info(
        "✅Saved suffixes to output file.", output_file=output_file, count=len(suffixes)
    )


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate suffixes for given prefixes using a language model"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()
    config = read_yaml_config(args.config_path)

    dataset = config.get("dataset", DATASET)
    model_name = config.get("model_name", MODEL_NAME)
    output_file = config.get("output_file", OUTPUT_FILE)
    num_suffixes = config.get("num_suffixes", NUM_SUFFIXES)
    seed = config.get("seed", SEED)
    save_golden_prefix = config.get("save_golden_prefix", SAVE_GOLDEN_PREFIX)
    toxicity_threshold = config.get("toxicity_threshold", TOXICITY_THRESHOLD)
    max_tokens = config.get("max_tokens", MAX_TOKENS)
    min_tokens = config.get("min_tokens", MIN_TOKENS)
    toxicity_keys = config.get("toxicity_keys", TOXICITY_KEYS)
    text_keys = config.get("text_keys", TEXT_KEYS)
    golden_prefix_keys = config.get("golden_prefix_keys", GOLDEN_PREFIX_KEYS)

    logger.info("Configuration:", config=config)

    random.seed(seed)

    # Calls the main processing function with the loaded configuration.
    process_dataset(
        dataset=dataset,
        model_name=model_name,
        output_file=output_file,
        num_suffixes=num_suffixes,
        save_golden_prefix=save_golden_prefix,
        toxicity_threshold=toxicity_threshold,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        text_keys=text_keys,
        toxicity_keys=toxicity_keys,
        golden_prefix_keys=golden_prefix_keys,
    )


if __name__ == "__main__":
    main()
