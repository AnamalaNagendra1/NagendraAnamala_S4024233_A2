# data/mini/make_mini_data.py
import json, random, pathlib
import numpy as np

base = pathlib.Path(__file__).parent
rng = random.Random(4024233)
np_rng = np.random.default_rng(4024233)

# 1) Make 120 synthetic docs so we can later select exactly 100 supports
doc_ids = [f"D{idx:06d}" for idx in range(1, 121)]
topics = ["policy","survey","method","retrieval","ranking","calibration","evidence","mapping","token","context"]

docs_path = base / "documents.jsonl"
with docs_path.open("w", encoding="utf-8") as f:
    for did in doc_ids:
        toks = [rng.choice(topics) for _ in range(rng.randint(20, 40))]
        text = f"{did} " + " ".join(toks)
        f.write(json.dumps({"id": did, "text": text}, ensure_ascii=False) + "\n")

# 2) One test question (MCQ A-D)
test_q = [{
    "question": "Which option best reflects calibrated retrieval-grounded answers for the survey?",
    "options": ["A","B","C","D"]
}]
with (base / "test_questions.json").open("w", encoding="utf-8") as f:
    json.dump(test_q, f, ensure_ascii=False, indent=2)

# 3) Tiny embeddings (optional for later)
vecs = np_rng.normal(size=(len(doc_ids), 8)).astype("float32")
np.savez(base / "id_to_embedding.npz", ids=np.array(doc_ids), vectors=vecs)

print("Wrote:", docs_path, "and test_questions.json and id_to_embedding.npz")
