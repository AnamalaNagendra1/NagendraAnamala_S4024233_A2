# index/build.py
import argparse, os, time
import yaml
from src_utils.io_ops import read_jsonl, write_json
from src_utils.sparse_bm25 import ToyBM25

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--api_key", default="", help="Accepted but unused")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    docs_path = cfg["data"]["documents_path"]
    run_dir   = cfg["outputs"]["run_dir"]
    k1 = cfg["retrieval"]["sparse"]["bm25_k1"]
    b  = cfg["retrieval"]["sparse"]["bm25_b"]

    rows = read_jsonl(docs_path)
    bm25 = ToyBM25(k1=k1, b=b)
    bm25.build(rows)

    stamp = str(int(time.time()))
    out_dir = os.path.join(run_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "N_docs": bm25.N,
        "avg_len": bm25.avg_len,
        "vocab_size": len(bm25.inv),
        "k1": k1,
        "b": b,
        "source_docs": os.path.abspath(docs_path),
    }
    write_json(os.path.join(out_dir, "sparse_index_meta.json"), meta)
    print(f"[index.build] Built sparse index from {bm25.N} docs → {out_dir}")

if __name__ == "__main__":
    main()
