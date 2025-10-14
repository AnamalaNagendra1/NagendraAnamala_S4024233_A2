# src_utils/io_ops.py
import json, os, random
from pathlib import Path

def read_jsonl(path):
    bag = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bag.append(json.loads(line))
    return bag

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path, obj):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def fix_seed(seed_val: int):
    random.seed(seed_val)
    try:
        import numpy as np
        np.random.seed(seed_val)
    except Exception:
        pass
