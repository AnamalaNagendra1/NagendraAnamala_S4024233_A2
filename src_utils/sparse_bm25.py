# src_utils/sparse_bm25.py
import re, math
from collections import defaultdict, Counter

_TOKEN_RX = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str):
    return [t.lower() for t in _TOKEN_RX.findall(text or "")]

def _safe_text(row: dict) -> str:
    """
    Prefer row['text'] if it exists and is a string.
    Otherwise, concatenate all string fields (except 'id').
    If a field is a list of strings, join them with spaces.
    """
    if isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"]

    parts = []
    for k, v in row.items():
        if k == "id":
            continue
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            parts.append(" ".join(v))
        # ignore dicts/other types
    return " ".join(parts)

class ToyBM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b =  b
        self.doc_store = {}         # doc_id -> text
        self.tf_maps   = {}         # doc_id -> Counter
        self.doc_len   = {}         # doc_id -> length
        self.inv       = defaultdict(set) # term -> {doc_id}
        self.avg_len   = 0.0
        self.N         = 0

    def build(self, rows):
        # rows: list of {"id": str, ...}
        self.doc_store.clear()
        self.tf_maps.clear()
        self.doc_len.clear()
        self.inv.clear()

        total_len = 0
        for r in rows:
            did = r.get("id") or r.get("doc_id") or r.get("_id")
            if not did:
                # skip rows without an id
                continue
            did = str(did)
            text = _safe_text(r)
            self.doc_store[did] = text
            toks = tokenize(text)
            tf = Counter(toks)
            self.tf_maps[did] = tf
            L = sum(tf.values())
            self.doc_len[did] = L
            total_len += L
            for t in tf:
                self.inv[t].add(did)

        self.N = len(self.doc_store)
        self.avg_len = (total_len / max(1, self.N))

    def _idf(self, term: str) -> float:
        n = len(self.inv.get(term, ()))
        if n == 0:
            return 0.0
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str):
        q_toks = tokenize(query)
        cand = set()
        for qt in set(q_toks):
            cand |= self.inv.get(qt, set())
        scores = []
        for did in cand:
            s = 0.0
            Ld = self.doc_len.get(did, 1) or 1
            for qt in q_toks:
                tf = self.tf_maps[did].get(qt, 0)
                if tf == 0:
                    continue
                idf = self._idf(qt)
                denom = tf + self.k1 * (1 - self.b + self.b * (Ld / (self.avg_len or 1.0)))
                s += idf * ((tf * (self.k1 + 1)) / denom)
            if s != 0.0:
                scores.append((did, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
