from datasets import load_dataset
import re, json, os

# Load dataset
ds = load_dataset("princeton-nlp/SWE-bench_Lite_oracle", split="test")

# Regex patterns
ISSUE_RE = re.compile(r"<issue>(.*?)</issue>", re.S)
CODE_RE  = re.compile(r"<code>(.*?)</code>", re.S)

def parse_entry(ex):
    text = ex["text"]
    issue = ISSUE_RE.search(text)
    code  = CODE_RE.search(text)
    return {
        "instance_id": ex["instance_id"],
        "repo": ex["repo"],
        "base_commit": ex["base_commit"],
        "issue": issue.group(1).strip() if issue else None,
        "code": code.group(1).strip() if code else None,
        "prompt": text.strip(),  # keep entire text for direct use
    }

parsed = [parse_entry(e) for e in ds]

# Save to disk
os.makedirs("swe-bench-lite", exist_ok=True)
with open("swe-bench-lite/train.jsonl", "w") as f:
    for ex in parsed:
        json.dump(ex, f)
        f.write("\n")

print(f"✅ Parsed {len(parsed)} SWE-bench Lite Oracle examples.")
