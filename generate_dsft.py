import argparse
import json
import os
import sys
from typing import Dict

import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

# === Config ===
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"  # or any other HF model
PREFIX_FILE = "prefixes_fineweb.jsonl"
OUTPUT_FILE = "sft_dataset.jsonl"
PREFIX_TOKENS = 64
BATCH_SIZE = 32  # adjust depending on GPU memory (A100 can handle 32–64 easily)
SAVE_EVERY = 500  # save every N generations


def read_yaml_config(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        print(f"Config file not found: {file_path}. Using default parameters.")
        return {}
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def generate_dsft(
    prefix_file: str,
    output_file: str,
    prefix_tokens: int,
    model_name: str,
    batch_size: int,
    save_every: int,
):
    # === Initialize vLLM model ===
    llm = LLM(model_name, dtype="bfloat16")  # or "float16" if you prefer
    sampling_params = SamplingParams(
        max_tokens=prefix_tokens,
        temperature=0.0,  # greedy decode
        top_p=1.0,
    )

    # === Verify prefix file exists ===
    if not os.path.exists(prefix_file):
        sys.exit(f"Prefix file not found: {prefix_file}")

    # === Load prefixes ===
    prefixes = []
    with open(prefix_file, "r", encoding="utf-8") as f:
        for line in f:
            prefixes.append(json.loads(line)["prefix"])

    # === Resume support ===
    done = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print(f"[Resume] Found {done} existing pairs, resuming from there.")

    # === Generation loop (batched) ===
    pairs_done = []
    for start in tqdm(
        range(done, len(prefixes), batch_size), desc="Generating suffixes"
    ):
        batch = prefixes[start : start + batch_size]

        # Run inference with vLLM
        outputs = llm.generate(batch, sampling_params, use_tqdm=False)

        for prefix, out in zip(batch, outputs):
            suffix = out.outputs[0].text.strip()

            pairs_done.append({"prefix": prefix, "suffix": suffix})

        # Save periodically
        if len(pairs_done) >= save_every or start + batch_size >= len(prefixes):
            if os.path.exists(output_file):
                mode = "a"
            else:
                mode = "w"
            with open(output_file, mode, encoding="utf-8") as f:
                for p in pairs_done:
                    f.write(json.dumps(p) + "\n")
            print(
                f"[Checkpoint] Saved {len(pairs_done)} new pairs (total {start+len(batch)})."
            )
            pairs_done = []

    print("✅ Done. All pairs saved to", output_file)


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

    prefix_file = config.get("prefix_file")
    output_file = config.get("output_file", OUTPUT_FILE)
    prefix_tokens = config.get("prefix_tokens", PREFIX_TOKENS)
    model_name = config.get("model_name", MODEL_NAME)
    batch_size = config.get("batch_size", BATCH_SIZE)
    save_every = config.get("save_every", SAVE_EVERY)

    print(f"Configuration: {config}")

    generate_dsft(
        prefix_file=prefix_file,
        output_file=output_file,
        prefix_tokens=prefix_tokens,
        model_name=model_name,
        batch_size=batch_size,
        save_every=save_every,
    )


if __name__ == "__main__":
    main()
