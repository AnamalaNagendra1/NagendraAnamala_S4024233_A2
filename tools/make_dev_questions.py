# tools/make_dev_questions.py
import json, pathlib

src = pathlib.Path("data/dev/dev.json")
dst = pathlib.Path("data/dev/dev_questions.json")

data = json.loads(src.read_text(encoding="utf-8"))

payload = data["root"] if isinstance(data, dict) and "root" in data else data
questions = []

if isinstance(payload, dict):
    for q, obj in payload.items():
        dist = obj.get("distribution", {})
        opts = list(dist.keys()) or ["A","B","C","D"]
        questions.append({"question": q, "options": opts})
elif isinstance(payload, list):
    for item in payload:
        if isinstance(item, dict):
            q = item.get("question")
            dist = item.get("distribution", {})
            opts = list(dist.keys()) or item.get("options", ["A","B","C","D"])
            if q:
                questions.append({"question": q, "options": opts})
else:
    raise SystemExit("Unsupported dev.json format")

dst.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {dst} with {len(questions)} questions")
