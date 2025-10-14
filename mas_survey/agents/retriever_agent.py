# mas_survey/agents/retriever_agent.py
from src_utils.sparse_bm25 import ToyBM25
from src_utils.io_ops import read_jsonl

def build_sparse_index(doc_path: str, k1: float = 1.2, b: float = 0.75):
    rows = read_jsonl(doc_path)
    engine = ToyBM25(k1=k1, b=b)
    engine.build(rows)
    return engine, [r["id"] for r in rows]

def retrieve(engine: ToyBM25, query: str, topk: int = 150):
    # returns a list of document IDs ranked by BM25
    return [did for did, _ in engine.score(query)[:topk]]
