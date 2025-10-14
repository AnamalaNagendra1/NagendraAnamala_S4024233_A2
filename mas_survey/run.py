# mas_survey/run.py
import argparse, os, time, csv, json
from decimal import Decimal, getcontext
getcontext().prec = 50

import yaml
from src_utils.io_ops import read_json
from mas_survey.agents.parser_agent import parse_question
from mas_survey.agents.retriever_agent import build_sparse_index, retrieve
from mas_survey.agents.responder_agent import pick_answer
from mas_survey.agents.verifier_agent import calibrate_distribution, enforce_unique_supports

def _uniform(options):
    n = len(options) or 1
    vals = [float(Decimal(1) / Decimal(n)) for _ in range(n)]
    vals[-1] += (1.0 - sum(vals))  # exact-sum correction
    return {options[i]: vals[i] for i in range(n)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--api_key", default="")  # accepted but unused
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Build sparse index
    engine, universe_ids = build_sparse_index(
        cfg["data"]["documents_path"],
        k1=cfg["retrieval"]["sparse"]["bm25_k1"],
        b=cfg["retrieval"]["sparse"]["bm25_b"],
    )

    supports_k = int(cfg["outputs"]["supports_k"])
    questions = read_json(cfg["data"]["test_questions_path"])  # list of {"question": "...", "options":[...]}

    rows = []
    for qobj in questions:
        qtext = qobj["question"]
        qinfo = parse_question(qtext)
        options = qobj.get("options") or qinfo["options"]

        # retrieval → candidate supports
        seeds = retrieve(engine, qinfo["normalized"], topk=cfg["retrieval"]["sparse"]["topk"])

        # answer → calibrated distribution
        best = pick_answer(qinfo["normalized"], options)
        dist = calibrate_distribution(best, options) if best in options else _uniform(options)

        # enforce exactly-K unique supports
        supports = enforce_unique_supports(seeds, supports_k, universe_ids)

        rows.append((qtext, json.dumps(dist, ensure_ascii=False), json.dumps(supports, ensure_ascii=False)))

    # write CSV
    run_dir = os.path.join(cfg["outputs"]["run_dir"], str(int(time.time())))
    os.makedirs(run_dir, exist_ok=True)
    out_csv = os.path.join(run_dir, "submission.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "distribution", "supports"])
        for r in rows:
            w.writerow(r)

    print(f"[mas_survey.run] Wrote {len(rows)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
