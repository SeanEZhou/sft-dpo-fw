import json, random, textwrap

file_path = "oracle_qwen7b_suffixes.jsonl"

# Step 1: count total lines
with open(file_path, "r") as f:
    total_lines = sum(1 for _ in f)

# Step 2: pick random line number
rand_line = random.randint(0, total_lines - 1)

# Step 3: load that entry
with open(file_path, "r") as f:
    for i, line in enumerate(f):
        if i == rand_line:
            entry = json.loads(line)
            break

# Step 4: print formatted
print("=" * 100)
print(f"🎯 Random Entry #{rand_line+1}/{total_lines}")
print(f"🧩 Instance ID: {entry['instance_id']}")
print("=" * 100)
print("\n📝 ISSUE:\n")
print(textwrap.indent(entry['issue'][:1500] + ("..." if len(entry['issue']) > 1500 else ""), "  "))
print("\n📜 SUFFIX (first 200 lines):\n")
suffix_lines = entry['suffix'].splitlines()
for i, line in enumerate(suffix_lines[:200]):
    print(f"{i+1:3d}: {line}")
if len(suffix_lines) > 200:
    print(f"... ({len(suffix_lines)-200} more lines)")
print("=" * 100)
