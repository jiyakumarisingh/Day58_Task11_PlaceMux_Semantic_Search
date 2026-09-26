import pandas as pd
from src.retrieval import SearchEngine
from src.evaluate import evaluate, split_labels, tune_weight

class FakeModel:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np
        out=[]
        for t in texts:
            t=t.lower()
            out.append([1.0 if "pipeline" in t else 0.0, 1.0 if "python" in t else 0.0, 1.0 if "sales" in t else 0.0])
        a=np.array(out,dtype="float32"); a/=((a*a).sum(1,keepdims=True)**0.5+1e-8); return a

def make_engine():
    return SearchEngine(["1","2","3"],["python data pipeline engineering","sales account management","data warehouse pipelines"],FakeModel())

def test_modes_return_results():
    e=make_engine()
    assert e.search("build data pipelines", "keyword", 2)["results"]
    assert e.search("build data pipelines", "semantic", 2)["results"]
    assert e.search("build data pipelines", "hybrid", 2)["results"]

def test_fallback():
    e=SearchEngine(["1","2"],["data pipelines","sales"],None)
    r=e.search("data pipelines", "semantic", 2)
    assert r["mode"] == "keyword_fallback"

def test_evaluation_and_tuning():
    labels=pd.DataFrame([
      ["q1","build pipelines","1",1],["q1","build pipelines","2",0],
      ["q2","sales person","2",1],["q2","sales person","1",0],
      ["q3","python pipeline","1",1],["q3","python pipeline","3",1]
    ],columns=["query_id","query","doc_id","label"])
    e=make_engine(); tr,te=split_labels(labels,test_size=.34,seed=1)
    assert evaluate(e,te,"keyword")["queries"] >= 1
    w, rows=tune_weight(e,tr)
    assert 0 <= w <= 1 and rows
