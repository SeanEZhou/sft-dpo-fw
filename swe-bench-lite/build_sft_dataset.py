import json, os
from tqdm import tqdm

TRAIN_FILE = "train.jsonl"             # has code + issue
PATCH_FILE = "oracle_qwen7b_suffixes.jsonl"          # has suffix (patch)
OUTPUT_FILE = "sft_investigator_dataset.jsonl"

# --- Load base train dataset ---
print(f"📂 Loading {TRAIN_FILE}")
train_data = {}
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        train_data[ex["instance_id"]] = {
            "code": ex["code"],
            "issue": ex["issue"],
        }

# --- Merge in patches ---
print(f"📂 Loading {PATCH_FILE}")
merged = []
missing, empty = 0, 0
with open(PATCH_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Merging"):
        ex = json.loads(line)
        task_id = ex["instance_id"]
        patch = ex.get("suffix", "").strip()
        if not patch:
            empty += 1
            continue
        if task_id not in train_data:
            missing += 1
            continue

        code = train_data[task_id]["code"]
        issue = train_data[task_id]["issue"]

        # Build training pair
        prompt = (
            "Given the following code base and the resulting patch, predict the task description.\n"
            "<code>\n"
            f"{code.strip()}\n"
            "</code>\n"
            "<patch>\n"
            f"{patch.strip()}\n"
            "</patch>\n"
        )
        label = issue.strip()

        merged.append({"instance_id": task_id, "prompt": prompt, "label": label})

# --- Save output ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in merged:
        f.write(json.dumps(ex) + "\n")

print(f"✅ Done. Saved {len(merged)} examples → {OUTPUT_FILE}")
print(f"⚠️ Skipped: {missing} missing ids, {empty} empty patches")
