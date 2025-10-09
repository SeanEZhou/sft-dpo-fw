import argparse
import gc
import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams

from logger import logger
from utils import read_yaml_config

# === Config ===
# Frank-Wolfe model.
INVESTIGATOR_MODEL = "/work7/sean/l8b_investigator_toxic_fw1_checkpoints/checkpoint-357"
# Base Language Model (p_m).
TARGET_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
# Model to penalize. Note: "" = no penalty
PENALIZE_MODEL = "/work7/sean/l8b_investigator_toxic_fw1_checkpoints/checkpoint-357"
# Weight for penalty term.
LAMBDA = 1.0

# File paths.
# Input suffix dataset.
DATA_FILE = "toxic_suffixes_l8b.jsonl"
# Output dataset of chosen/rejected pairs.
OUTPUT_FILE = "toxic_fw2_l8b.jsonl"
# Progress file.
PROGRESS_FILE = "toxic_l8b_fw2_progress.json"


# Generation and scoring parameters.
# Candidates and pairs.
NUM_CANDIDATES = 8
PAIRS_PER_SUFFIX = 2
# Generation parameters.
MAX_NEW_TOKENS = 64
# Batch size for suffixes.
BATCH_SIZE_SUFFIXES = 8  # suffixes processed in parallel
# Minimum prefix length.
MIN_PREFIX_LEN = 10  # minimum prefix len we'll keep in dataset
# Minimum score difference.
MINIMUM_SCORE_DIFF = 0.05
# Devices.
# Device for Investigator.
DEVICE_INV = "cuda:0"
# Device for Target Model for scoring proposed prefixes.
DEVICE_TARGET = "cuda:0"
# Device for Penalizing Model for penalizing repeated proposals.
DEVICE_PENALIZE = "cuda:0"

QUANTISE_TARGET = False  # whether to load target model with 8-bit quantization


def generate_suffixes(
    investigator_model_name,
    target_model_name,
    penalise_model_name,
    lambda_,
    data_file,
    output_file,
    progress_file,
    num_candidates,
    pairs_per_suffix,
    max_new_tokens,
    batch_size_suffixes,
    min_prefix_len,
    minimum_score_diff,
    device_inv,
    device_target,
    device_penalize,
    quantise_target,
):
    # === Load tokenizer ===
    tok = AutoTokenizer.from_pretrained(target_model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # === Load models ===
    llm = LLM(
        model=investigator_model_name, dtype="bfloat16"
    )  # investigator (prefix generator)

    bnb_config = BitsAndBytesConfig(
        # Load the model with 4-bit quantization
        load_in_4bit=True,
        # Use double quantization
        bnb_4bit_use_double_quant=True,
        # Use 4-bit Normal Float for storing the base model weights in GPU memory
        bnb_4bit_quant_type="nf4",
        # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    target_model = (
        AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config if quantise_target else None,
        )
        .to(device_target)
        .eval()
    )

    penalize_model = None
    if penalise_model_name:
        penalize_model = (
            AutoModelForCausalLM.from_pretrained(penalise_model_name)
            .to(device_penalize)
            .eval()
        )

    # === Load suffix dataset ===
    dataset = load_dataset("json", data_files=data_file, split="train")
    suffixes = [ex["suffix"] for ex in dataset]

    # === Resume support ===
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
        start_idx = progress.get("last_suffix_idx", 0) + 1
    else:
        start_idx = 0

    logger.info(
        "Starting from suffix.", start_idx=start_idx, total_suffixes=len(suffixes)
    )

    # === Scoring functions ===
    @torch.inference_mode()
    def batch_score_suffix_given_prefix(prefixes, suffixes):
        """Computes avg log p_m(y|x) for a batch of (prefix, suffix)."""
        texts = [f"{p} {s}" for p, s in zip(prefixes, suffixes)]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(
            device_target
        )
        prefix_lens = [
            len(tok(p, add_special_tokens=False)["input_ids"]) for p in prefixes
        ]

        outputs = target_model(enc.input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = enc.input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        scores = []
        for i, plen in enumerate(prefix_lens):
            mask = torch.arange(labels.shape[1], device=device_target) >= (plen - 1)
            total = (chosen[i] * mask).sum().item()
            count = mask.sum().item()
            scores.append(total / max(count, 1))
        return scores

    @torch.inference_mode()
    def batch_score_prefix_given_suffix(prefixes, suffixes):
        """Computes avg log p_prev(x|y) for a batch."""
        if not penalize_model:
            return [0.0] * len(prefixes)

        texts = [f"<suffix> {s} <prefix> {p}" for p, s in zip(prefixes, suffixes)]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(
            device_penalize
        )
        prefix_starts = [
            len(tok(f"<suffix> {s} <prefix>", add_special_tokens=False)["input_ids"])
            for s in suffixes
        ]

        outputs = penalize_model(enc.input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = enc.input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        scores = []
        for i, pstart in enumerate(prefix_starts):
            mask = torch.arange(labels.shape[1], device=device_penalize) >= (pstart - 1)
            total = (chosen[i] * mask).sum().item()
            count = mask.sum().item()
            scores.append(total / max(count, 1))
        return scores

    def batch_score_total(prefixes, suffixes):
        base_scores = batch_score_suffix_given_prefix(prefixes, suffixes)
        penalties = batch_score_prefix_given_suffix(prefixes, suffixes)
        return [b - lambda_ * p for b, p in zip(base_scores, penalties)]

    # === Main loop ===
    with open(output_file, "a") as fout:
        for batch_start in tqdm(
            range(start_idx, len(suffixes), batch_size_suffixes),
            desc="Processing suffixes",
        ):
            batch_suffixes = suffixes[batch_start : batch_start + batch_size_suffixes]

            # 1. Generate candidate prefixes for all suffixes with vLLM
            prompts = [f"<suffix> {s} <prefix>" for s in batch_suffixes]
            sampling_params = SamplingParams(
                n=num_candidates, max_tokens=max_new_tokens, temperature=1.1, top_p=0.95
            )
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            # 2. For each suffix in the batch
            for j, sfx in enumerate(batch_suffixes):
                candidates = set()
                for o in outputs[j].outputs:
                    decoded = o.text.strip()
                    prefix = decoded.split("<prefix>")[-1].strip()
                    if len(prefix) >= min_prefix_len:
                        candidates.add(prefix)

                # if fewer than 2 candidates, skip this suffix
                if len(candidates) < 2:
                    continue

                # 3. Score candidates
                prefix_list = list(candidates)
                suffix_list = [sfx] * len(prefix_list)
                scores = batch_score_total(prefix_list, suffix_list)
                scored = list(zip(prefix_list, scores))
                scored.sort(key=lambda x: x[1], reverse=True)

                # 4. Build preference pairs
                pairs = []
                if (
                    len(scored) >= 2
                    and abs(scored[0][1] - scored[-1][1]) >= minimum_score_diff
                ):
                    # always include best vs worst
                    pairs.append(
                        {
                            "chosen": scored[0][0],
                            "rejected": scored[-1][0],
                            "suffix": sfx,
                        }
                    )
                else:
                    continue

                # 5. Write pairs
                for p in pairs:
                    fout.write(json.dumps(p) + "\n")
                fout.flush()

                # 6. Update progress
                with open(progress_file, "w") as f:
                    json.dump(
                        {
                            "last_suffix_idx": batch_start + j,
                        },
                        f,
                    )

            torch.cuda.empty_cache()
            gc.collect()

    logger.info("✅ FW/DPO dataset generation complete.")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate a dataset for Frank-Wolfe/DPO training."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()
    config = read_yaml_config(args.config_path)

    investigator_model_name = config.get("investigator_model_name", INVESTIGATOR_MODEL)
    target_model_name = config.get("target_model_name", TARGET_MODEL_NAME)
    penalise_model_name = config.get("penalise_model_name", PENALIZE_MODEL)
    lambda_ = config.get("lambda", LAMBDA)
    data_file = config.get("data_file", DATA_FILE)
    output_file = config.get("output_file", OUTPUT_FILE)
    progress_file = config.get("progress_file", PROGRESS_FILE)
    num_candidates = config.get("num_candidates", NUM_CANDIDATES)
    pairs_per_suffix = config.get("pairs_per_suffix", PAIRS_PER_SUFFIX)
    max_new_tokens = config.get("max_new_tokens", MAX_NEW_TOKENS)
    batch_size_suffixes = config.get("batch_size_suffixes", BATCH_SIZE_SUFFIXES)
    min_prefix_len = config.get("min_prefix_len", MIN_PREFIX_LEN)
    minimum_score_diff = config.get("minimum_score_diff", MINIMUM_SCORE_DIFF)
    device_inv = config.get("device_inv", DEVICE_INV)
    device_target = config.get("device_target", DEVICE_TARGET)
    device_penalize = config.get("device_penalize", DEVICE_PENALIZE)
    quantise_target = config.get("quantise_target", QUANTISE_TARGET)

    visible_devices = [device_inv, device_target, device_penalize]
    visible_devices_idxs = []
    for i in range(len(visible_devices)):
        if visible_devices[i].startswith("cuda:"):
            visible_devices_idxs.append(visible_devices[i].replace("cuda:", ""))
        elif visible_devices[i].startswith("cpu"):
            continue
        else:
            logger.error(
                "Invalid device specified. Use 'cuda:<id>' format.",
                device=visible_devices[i],
            )
            raise ValueError("Invalid device format.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(set(visible_devices_idxs))
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"DATA_FILE not found: {data_file}. Check path and filename."
        )

    logger.info("Configuration loaded.", config=config)

    generate_suffixes(
        investigator_model_name,
        target_model_name,
        penalise_model_name,
        lambda_,
        data_file,
        output_file,
        progress_file,
        num_candidates,
        pairs_per_suffix,
        max_new_tokens,
        batch_size_suffixes,
        min_prefix_len,
        minimum_score_diff,
        device_inv,
        device_target,
        device_penalize,
        quantise_target,
    )


if __name__ == "__main__":
    main()
