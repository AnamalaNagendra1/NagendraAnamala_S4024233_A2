# tools/quick_check.py
import csv, json, glob

p = sorted(glob.glob("runs/*/submission.csv"))[-1]
with open(p, "r", encoding="utf-8", newline="") as f:
    row = next(csv.DictReader(f))

dist = json.loads(row["distribution"])
supports = json.loads(row["supports"])

print("file:", p)
print("prob sum =", sum(dist.values()))
print("supports count =", len(supports))
print("unique supports =", len(set(supports)))
