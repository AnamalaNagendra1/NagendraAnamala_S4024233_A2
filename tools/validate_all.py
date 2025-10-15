# tools/validate_all.py
import csv, json, glob, sys

def latest_csv():
    paths = sorted(glob.glob("runs/*/submission.csv"))
    if not paths:
        print("No runs/*/submission.csv found"); sys.exit(1)
    return paths[-1]

def load_doc_ids():
    ids = set()
    with open("data/dev/documents.jsonl","r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            did = row.get("id") or row.get("doc_id") or row.get("_id")
            if did is not None:
                ids.add(str(did))
    return ids

def load_questions():
    with open("data/dev/dev_questions.json","r",encoding="utf-8") as f:
        arr = json.load(f)
    return {x["question"]: x.get("options",[]) for x in arr}

def main():
    csv_path = latest_csv()
    doc_ids = load_doc_ids()
    qmap = load_questions()
    tol = 1e-6
    errors = []
    seen = set()
    n = 0

    with open(csv_path,"r",encoding="utf-8",newline="") as f:
        rdr = csv.DictReader(f)
        for i,row in enumerate(rdr, start=2):  # header is line 1
            n += 1
            q = row.get("question","")
            if q not in qmap:
                errors.append(f"Row {i}: question not in dev set")
            if q in seen:
                errors.append(f"Row {i}: duplicate question")
            seen.add(q)

            # distribution
            try:
                dist = json.loads(row["distribution"])
                s = sum(float(v) for v in dist.values())
                if abs(s - 1.0) > tol:
                    errors.append(f"Row {i}: prob sum {s} not within ±{tol}")
                if any(float(v) < 0.0 for v in dist.values()):
                    errors.append(f"Row {i}: negative probability found")
                opts = set(qmap.get(q, []))
                if opts and not set(dist.keys()).issubset(opts):
                    errors.append(f"Row {i}: distribution keys not subset of options")
            except Exception as e:
                errors.append(f"Row {i}: distribution parse error: {e}")

            # supports
            try:
                supports = json.loads(row["supports"])
                if not isinstance(supports, list):
                    errors.append(f"Row {i}: supports is not a list")
                elif len(supports) != 100:
                    errors.append(f"Row {i}: supports count {len(supports)} != 100")
                elif len(set(supports)) != len(supports):
                    errors.append(f"Row {i}: supports contain duplicates")
                missing = [s for s in supports if s not in doc_ids]
                if missing:
                    errors.append(f"Row {i}: supports not in corpus, e.g. {missing[:3]}")
            except Exception as e:
                errors.append(f"Row {i}: supports parse error: {e}")

    if n != len(qmap):
        errors.append(f"Row count {n} != number of dev questions {len(qmap)}")

    if errors:
        print("VALIDATION: FAIL")
        for e in errors[:50]:
            print(" -", e)
        if len(errors) > 50:
            print(f" ... and {len(errors)-50} more")
        sys.exit(1)
    else:
        print("VALIDATION: OK")
        print(f"Rows: {n} · Questions matched: {len(seen)} · Supports per row: 100")

if __name__ == "__main__":
    main()
