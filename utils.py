import csv
import json
import os
from typing import Dict

import yaml
from transformers import TrainerCallback


def read_jsonl_file(jsonl_path: str):
    with open(jsonl_path, "r") as f:
        for line in f:
            yield json.loads(line)


def read_yaml_config(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        print(f"Config file not found: {file_path}. Using default parameters.")
        return {}
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def read_json_file(jsonl_path: str):
    if jsonl_path.endswith(".jsonl"):
        with open(jsonl_path, "r") as f:
            for line in f:
                yield json.loads(line)
    elif jsonl_path.endswith(".json"):
        with open(jsonl_path, "r") as f:
            data = json.load(f)
            for item in data:
                yield item
    else:
        raise ValueError("File must be .json or .jsonl")


def get_nested(d, keys):
    for key in keys:
        d = d[key]
    return d


class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.header_written = False

    def _write_logs(self, logs):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        write_header = not os.path.exists(self.log_file) or not self.header_written
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=logs.keys())
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(logs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"] = state.global_step
            logs["type"] = "train"
            self._write_logs(logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            metrics["step"] = state.global_step
            metrics["type"] = "eval"
            self._write_logs(metrics)
