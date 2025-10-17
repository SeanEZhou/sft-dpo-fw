import os, json
from tqdm import tqdm
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
INPUT_FILE = "train.jsonl"
OUTPUT_FILE = "oracle_qwen7b_suffixes.jsonl"
BATCH_SIZE = 1
MAX_TOKENS = 2048
TEMPERATURE = 0.2
SAVE_EVERY = 20
MAX_CONTEXT = 32000   # safety limit for Qwen 2.5

def main():
    print(f"🚀 Loading {MODEL_NAME}")
    llm = LLM(model=MODEL_NAME, dtype="bfloat16")
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        stop=["</patch>", "</s>"],
    )

    # Load dataset
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    print(f"Loaded {len(entries)} tasks.")

    done = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print(f"[Resume] Skipping {done} already completed.")
    entries = entries[done:]

    buffer = []
    for i, ex in enumerate(tqdm(entries, desc="Generating")):
        user_prompt = f"<|user|>\n{ex['prompt']}\n<|assistant|>\n"

        # --- Check token length ---
        toks = llm.get_tokenizer().encode(user_prompt)
        if len(toks) > MAX_CONTEXT:
            print(f"⚠️  {ex['instance_id']} too long ({len(toks)} > {MAX_CONTEXT}) → truncating.")
            # keep last portion of code (often more relevant)
            user_prompt = llm.get_tokenizer().decode(toks[-MAX_CONTEXT:])

        # --- Show debug for first few ---
        if i < 2:
            print("\n" + "="*100)
            print(f"🧩 {ex['instance_id']}  ({len(toks)} tokens before trunc)")
            print(user_prompt[:800])
            print("="*100)

        try:
            outputs = llm.generate([user_prompt], sampling, use_tqdm=False)
            suffix = outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"❌ Generation failed for {ex['instance_id']}: {e}")
            suffix = ""

        print(f"[{ex['instance_id']}] Generated {len(suffix)} chars")
        if not suffix:
            print(f"⚠️  Empty or failed generation for {ex['instance_id']}")

        buffer.append({
            "instance_id": ex["instance_id"],
            "issue": ex["issue"],
            "suffix": suffix,
        })

        # Save periodically
        if len(buffer) >= SAVE_EVERY or i == len(entries) - 1:
            mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
            with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
                for item in buffer:
                    f.write(json.dumps(item) + "\n")
            print(f"[Checkpoint] saved {len(buffer)} entries → {OUTPUT_FILE}")
            buffer.clear()

    print("✅ Done safely.")

if __name__ == "__main__":
    main()
