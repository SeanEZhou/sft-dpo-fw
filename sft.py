import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from utils import read_yaml_config

# === Config ===
CUDA_VISIBLE_DEVICES = "0"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
DATA_FILE = "fineweb_train.jsonl"
OUTPUT_DIR = "/work7/sean/l8b_investigator_sft_checkpoints"
MAX_LENGTH = 128  # suffix (64) + prefix (64)
BATCH_SIZE = 16  # adjust for GPU VRAM
EPOCHS = 3
SAVE_STEPS = 250
LOG_STEPS = 50
METRICS_FILE = "train_metrics.jsonl"


def sft(
    model_name: str,
    data_file: str,
    output_dir: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    save_steps: int,
    log_steps: int,
    metrics_file: str,
):
    """Supervised Fine-Tuning (SFT) training script."""

    os.makedirs(output_dir, exist_ok=True)

    # === Load dataset ===
    dataset = load_dataset("json", data_files=data_file, split="train")

    # === Tokenizer ===
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # === Preprocess: input = suffix, label = prefix ===
    def preprocess(example):
        text = f"<suffix> {example['suffix']} <prefix> {example['prefix']}"
        tokens = tok(text)
        return tokens

    train_ds = dataset.map(preprocess, remove_columns=["prefix", "suffix"])

    # === Data collator ===
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # === Model ===
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # === Training args ===
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        do_eval=False,
        eval_steps=log_steps,
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_safetensors=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_steps=50,
        learning_rate=5e-5,
        bf16=True,
        report_to="none",
    )

    # === Metrics ===
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).item()

        ppl = np.exp(loss)
        return {"loss": loss, "perplexity": ppl}

    # === Callback to save metrics ===
    class SaveMetricsCallback(TrainerCallback):
        def __init__(self, path=metrics_file):
            self.path = path
            if not os.path.exists(self.path):
                open(self.path, "w").close()

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None:
                with open(self.path, "a") as f:
                    f.write(json.dumps({"step": state.global_step, **metrics}) + "\n")

    # === Trainer ===
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(SaveMetricsCallback(metrics_file))

    # === Train (resumable) ===
    latest_ckpt = None
    if os.path.isdir(output_dir):
        # look for subdirectories named "checkpoint-*"
        subdirs = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
        ]
        if subdirs:
            latest_ckpt = max(subdirs, key=os.path.getmtime)  # newest checkpoint

    trainer.train(resume_from_checkpoint=latest_ckpt)


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
    data_file = config.get("data_file", DATA_FILE)
    output_dir = config.get("output_dir", OUTPUT_DIR)
    max_length = config.get("max_length", MAX_LENGTH)
    batch_size = config.get("batch_size", BATCH_SIZE)
    epochs = config.get("epochs", EPOCHS)
    save_steps = config.get("save_steps", SAVE_STEPS)
    log_steps = config.get("log_steps", LOG_STEPS)
    metrics_file = config.get("metrics_file", METRICS_FILE)
    cuda_visible_devices = config.get("cuda_visible_devices", CUDA_VISIBLE_DEVICES)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    sft(
        model_name,
        data_file,
        output_dir,
        max_length,
        batch_size,
        epochs,
        save_steps,
        log_steps,
        metrics_file,
    )


if __name__ == "__main__":
    main()
