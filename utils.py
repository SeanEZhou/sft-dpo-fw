import json
import os
from typing import Dict

import yaml


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
    with open(jsonl_path, "r") as f:
        data = json.load(f)
        for item in data:
            yield item
