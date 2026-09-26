from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
from src.data import load_documents, load_labels
from src.evaluate import split_labels, evaluate, tune_weight
from src.model import load_embedding_model
from src.retrieval import SearchEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--data", default="data")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    data = Path(args.data)
    out = Path("logs"); out.mkdir(exist_ok=True)
    resumes, jobs = load_documents(data)
    labels = load_labels(data)
    docs = pd.concat([
        resumes.rename(columns={"_text": "text"})[["doc_id", "text"]],
        jobs.rename(columns={"_text": "text"})[["doc_id", "text"]]
    ], ignore_index=True).drop_duplicates("doc_id")
    print(f"Resumes: {len(resumes)} | JDs: {len(jobs)} | Label rows: {len(labels)} | Queries: {labels.query_id.nunique()}")
    print("Loading embedding model...")
    model = load_embedding_model(args.model)
    engine = SearchEngine(docs.doc_id, docs.text, model=model)
    train, test = split_labels(labels)
    print(f"Train queries: {train.query_id.nunique()} | Held-out queries: {test.query_id.nunique()}")
    keyword = evaluate(engine, test, "keyword", k=args.k)
    semantic = evaluate(engine, test, "semantic", k=args.k)
    best_weight, tuning = tune_weight(engine, train, k=args.k)
    hybrid = evaluate(engine, test, "hybrid", weight=best_weight, k=args.k)
    print(json.dumps({"keyword": keyword, "semantic": semantic, "hybrid": hybrid, "tuned_semantic_weight": best_weight}, indent=2))
    query = "someone who can build data pipelines"
    print("\nWORKED EXAMPLE")
    for mode in ["keyword", "semantic", "hybrid"]:
        result = engine.search(query, mode=mode, top_k=3, semantic_weight=best_weight)
        print(f"\n{mode.upper()}")
        for r in result["results"]:
            print(f"{r['doc_id']} | score={r['score']} | {r['reason']}")
    payload = {"model": args.model, "k": args.k, "train_queries": int(train.query_id.nunique()), "test_queries": int(test.query_id.nunique()), "tuned_semantic_weight": best_weight, "keyword": keyword, "semantic": semantic, "hybrid": hybrid, "tuning": tuning, "worked_example_query": query}
    (out / "day58_experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nExperiment log: logs/day58_experiment.json")
    print("Fallback check: instantiate SearchEngine with model=None; semantic requests automatically use keyword retrieval.")

if __name__ == "__main__":
    main()
