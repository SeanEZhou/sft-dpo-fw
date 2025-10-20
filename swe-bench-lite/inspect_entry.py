import json, random, textwrap

file_path = "train.jsonl"

# Count lines
with open(file_path, "r") as f:
    total_lines = sum(1 for _ in f)

# Pick random line
rand_line = random.randint(0, total_lines - 1)

# Read and parse that entry
with open(file_path, "r") as f:
    for i, line in enumerate(f):
        if i == rand_line:
            entry = json.loads(line)
            break

print("=" * 120)
print(f"🎯 Random Entry #{rand_line+1}/{total_lines}")
print(f"🧩 Instance ID: {entry['instance_id']}")
print(f"📦 Repo: {entry['repo']}")
print(f"🔗 Base Commit: {entry['base_commit']}")
print("=" * 120)

# === ISSUE ===
print("\n📝 ISSUE (first 1000 chars):\n")
issue_preview = entry.get("issue", "")[:1000]
print(textwrap.indent(issue_preview + ("..." if len(entry.get('issue', '')) > 1000 else ""), "  "))

# === CODE ===
print("\n💻 CODE (first 200 lines):\n")
lines = entry.get("code", "").splitlines()
for i, line in enumerate(lines[:200]):
    print(f"{i+1:3d}: {line}")
if len(lines) > 200:
    print(f"... ({len(lines)-200} more lines)")

# === PROMPT ===
print("\n📜 FULL PROMPT (first 200 lines):\n")
prompt_lines = entry.get("prompt", "").splitlines()
for i, line in enumerate(prompt_lines[:200]):
    print(f"{i+1:3d}: {line}")
if len(prompt_lines) > 200:
    print(f"... ({len(prompt_lines)-200} more lines)")

print("=" * 120)
