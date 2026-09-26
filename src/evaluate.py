from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_labels(df, test_size=0.3, seed=42):
    groups = df["query_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(df, groups=groups))
    return df.iloc[tr].copy(), df.iloc[te].copy()


def _metrics_for_query(ranked, relevant, k):
    ranked = list(ranked)[:k]
    rel = set(map(str, relevant))
    hits = [1 if str(x) in rel else 0 for x in ranked]
    precision = sum(hits) / k
    recall = sum(hits) / max(len(rel), 1)
    rr = 0.0
    for idx, h in enumerate(hits, 1):
        if h:
            rr = 1.0 / idx
            break
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = min(len(rel), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal))
    ndcg = dcg / idcg if idcg else 0.0
    return precision, recall, rr, ndcg


def evaluate(engine, labels, mode, weight=0.7, k=5):
    rows = []
    for qid, g in labels.groupby("query_id"):
        query = str(g.iloc[0]["query"])
        relevant = g.loc[g.label > 0, "doc_id"].astype(str).tolist()
        if not relevant:
            continue
        result = engine.search(query, mode=mode, top_k=k, semantic_weight=weight)
        ranked = [r["doc_id"] for r in result["results"]]
        p, rec, rr, ndcg = _metrics_for_query(ranked, relevant, k)
        rows.append({"query_id": qid, "precision": p, "recall": rec, "mrr": rr, "ndcg": ndcg})
    if not rows:
        return {"queries": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
    x = pd.DataFrame(rows)
    return {"queries": len(x), "precision_at_k": float(x.precision.mean()), "recall_at_k": float(x.recall.mean()), "mrr": float(x.mrr.mean()), "ndcg_at_k": float(x.ndcg.mean())}


def tune_weight(engine, train_labels, weights=None, k=5):
    weights = weights or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []
    for w in weights:
        m = evaluate(engine, train_labels, "hybrid", weight=w, k=k)
        rows.append({"semantic_weight": w, **m})
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["mrr", "ndcg_at_k", "recall_at_k"], ascending=False).iloc[0]
    return float(best.semantic_weight), frame.to_dict(orient="records")
