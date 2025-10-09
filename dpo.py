import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from logger import logger
from utils import CSVLoggerCallback, read_yaml_config

# === Config ===
MODEL_NAME = "/work7/sean/l8b_investigator_sft_checkpoints/checkpoint-846"
REF_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
DATA_FILE = "fineweb_dpo_l8b.jsonl"
OUTPUT_DIR = "/work7/sean/l8b_investigator_dpo_checkpoints"

DEVICE = "cuda"
CUDA_VISIBLE_DEVICES = "4,5,7"


def dpo(model_name: str, ref_model_name: str, data_file: str, output_dir: str):
    """Trains a model using Direct Preference Optimization (DPO) on a given dataset."""
    # === Load dataset ===
    dataset = load_dataset("json", data_files=data_file, split="train")

    # Ensure 'prompt' column exists
    def add_prompt(example):
        if "prompt" not in example:
            example["prompt"] = f"Suffix:\n{example['suffix']}\nPrefix:\n"
        return example

    dataset = dataset.map(add_prompt)

    # === Load tokenizer ===
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # === Load models ===
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # === DPO Config ===
    dpo_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=5e-6,
        beta=0.1,
        logging_steps=50,
        save_steps=200,
        eval_steps=None,
        eval_strategy="no",
        warmup_steps=50,
        bf16=True,
        save_total_limit=3,
        remove_unused_columns=False,  # ✅ critical for custom schema
        report_to=[],  # disable wandb unless you want logging
        save_safetensors=True,  # ✅ ensure safetensors format
    )

    # === DPO Trainer ===
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tok,
    )

    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    trainer.add_callback(CSVLoggerCallback(metrics_file))

    # === Resume logic ===
    latest_ckpt = None
    if os.path.isdir(output_dir):
        ckpts = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
        ]
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getmtime)

    train_result = trainer.train(
        resume_from_checkpoint=latest_ckpt if latest_ckpt else None
    )

    # === Print summary of last metrics ===
    logger.info("Training finished.", train_result=train_result)


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

    model_name = config.get("model_name", MODEL_NAME)
    ref_model_name = config.get("ref_model_name", REF_MODEL_NAME)
    data_file = config.get("data_file", DATA_FILE)
    output_dir = config.get("output_dir", OUTPUT_DIR)
    device = config.get("device", DEVICE)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.get(
        "cuda_visible_devices", CUDA_VISIBLE_DEVICES
    )

    logger.info("Configuration", config=config)

    dpo(
        model_name=model_name,
        ref_model_name=ref_model_name,
        data_file=data_file,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
