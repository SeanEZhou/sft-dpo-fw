import argparse
import gc
import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from logger import logger
from utils import read_yaml_config

DATA_FILE = "toxic_suffixes_l8b.jsonl"  # suffix dataset
OUTPUT_FILE = "toxic_fw2_l8b.jsonl"  # chosen/rejected pairs
PROGRESS_FILE = "toxic_l8b_fw2_progress.json"

NUM_CANDIDATES = 8  # candidate prefixes per suffix
PAIRS_PER_SUFFIX = 2  # pairs to save per suffix
MAX_NEW_TOKENS = 64
BATCH_SIZE_SUFFIXES = 8  # suffixes processed in parallel
MIN_PREFIX_LEN = 10  # minimum prefix len we'll keep in dataset
MINIMUM_SCORE_DIFF = 0.05  # minimum score diff between pairs that we'll accept
DEVICE_INV = "cuda:0"  # device for investigator
DEVICE_TARGET = "cuda:0"  # device for target model, for scoring proposed prefixes
DEVICE_PENALIZE = (
    "cuda:0"  # device for the previous fw model, for penalizing repeated proposals
)


def generate_suffixes(
    INVESTIGATOR_MODEL,
    TARGET_MODEL_NAME,
    PENALIZE_MODEL,
    DATA_FILE,
    OUTPUT_FILE,
    PROGRESS_FILE,
    NUM_CANDIDATES,
    PAIRS_PER_SUFFIX,
    MAX_NEW_TOKENS,
    BATCH_SIZE_SUFFIXES,
    MIN_PREFIX_LEN,
    MINIMUM_SCORE_DIFF,
    DEVICE_INV,
    DEVICE_TARGET,
    DEVICE_PENALIZE,
):
    # === Load tokenizer ===
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # === Load models ===
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "0,1,2"  # controls what gpu the investigator uses
    )
    llm = LLM(
        model=INVESTIGATOR_MODEL, dtype="bfloat16", tensor_parallel_size=1
    )  # investigator (prefix generator)
    target_model = (
        AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_NAME, torch_dtype=torch.bfloat16
        )
        .to(DEVICE_TARGET)
        .eval()
    )

    penalize_model = None
    if PENALIZE_MODEL:
        penalize_model = (
            AutoModelForCausalLM.from_pretrained(PENALIZE_MODEL)
            .to(DEVICE_PENALIZE)
            .eval()
        )

    # === Load suffix dataset ===
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    suffixes = [ex["suffix"] for ex in dataset]

    # === Resume support ===
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
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
            DEVICE_TARGET
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
            mask = torch.arange(labels.shape[1], device=DEVICE_TARGET) >= (plen - 1)
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
            DEVICE_PENALIZE
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
            mask = torch.arange(labels.shape[1], device=DEVICE_PENALIZE) >= (pstart - 1)
            total = (chosen[i] * mask).sum().item()
            count = mask.sum().item()
            scores.append(total / max(count, 1))
        return scores

    def batch_score_total(prefixes, suffixes):
        base_scores = batch_score_suffix_given_prefix(prefixes, suffixes)
        penalties = batch_score_prefix_given_suffix(prefixes, suffixes)
        return [b - LAMBDA * p for b, p in zip(base_scores, penalties)]

    # === Main loop ===
    with open(OUTPUT_FILE, "a") as fout:
        for batch_start in tqdm(
            range(start_idx, len(suffixes), BATCH_SIZE_SUFFIXES),
            desc="Processing suffixes",
        ):
            batch_suffixes = suffixes[batch_start : batch_start + BATCH_SIZE_SUFFIXES]

            # 1. Generate candidate prefixes for all suffixes with vLLM
            prompts = [f"<suffix> {s} <prefix>" for s in batch_suffixes]
            sampling_params = SamplingParams(
                n=NUM_CANDIDATES, max_tokens=MAX_NEW_TOKENS, temperature=1.1, top_p=0.95
            )
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            # 2. For each suffix in the batch
            for j, sfx in enumerate(batch_suffixes):
                candidates = set()
                for o in outputs[j].outputs:
                    decoded = o.text.strip()
                    prefix = decoded.split("<prefix>")[-1].strip()
                    if len(prefix) >= MIN_PREFIX_LEN:
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
                    and abs(scored[0][1] - scored[-1][1]) >= MINIMUM_SCORE_DIFF
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
                with open(PROGRESS_FILE, "w") as f:
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
        description="Generate suffixes for given prefixes using a language model"
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

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"DATA_FILE not found: {data_file}. Check path and filename."
        )

    logger.info("Configuration loaded.", config=config)

    generate_suffixes(
        investigator_model_name,
        target_model_name,
        penalise_model_name,
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
    )


if __name__ == "__main__":
    main()
