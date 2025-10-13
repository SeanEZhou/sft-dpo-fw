import json

def read_jsonl(jsonl_path="artifacts/malicious_responses.jsonl"):
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            json_line = json.loads(line)
            yield json_line

def retrieve_code_snippet(response:str):
    start_marker = "```python"
    end_marker = "```"
    start_index = response.find(start_marker)
    if start_index == -1:
        return None
    start_index += len(start_marker)
    end_index = response.find(end_marker, start_index)
    if end_index == -1:
        return None
    code_snippet = response[start_index:end_index].strip()
    return code_snippet

if __name__ == "__main__":
    jsonl_paths = [
        "QA-FlowGPT-1.jsonl",
        "QA-FlowGPT-2.jsonl",
        "QA-FlowGPT-3.jsonl",
        "QA-Poe-1.jsonl",
        "QA-Poe-2.jsonl",
        "QA-Poe-3.jsonl",
    ]

    for jsonl_path in jsonl_paths:
        for item in read_jsonl(jsonl_path=jsonl_path):
            # print(item.get("response"))
            snippet = retrieve_code_snippet(item.get("response"))
            query = item.get("query")

            # print(retrieve_code_snippet(item.get("response")))
            if snippet:

                print("----- Extracted Code Snippet -----")
                print(snippet)

                with open("malicious_code.jsonl", "a") as f:
                    f.write(json.dumps({"code": snippet, "query": query}) + "\n")


    # Remove duplicates
    unique_snippets = set()
    with open("malicious_code.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            unique_snippets.add(line.strip())
    with open("malicious_code.jsonl", "w") as f:
        for snippet in unique_snippets:
            f.write(snippet + "\n")