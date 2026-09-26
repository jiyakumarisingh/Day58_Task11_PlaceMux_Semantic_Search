# PlaceMux — Day 58 / Task 11
## Semantic Search, Vector Retrieval & Hybrid Search

### Objective
Ship semantic search over resumes and JDs using embeddings and vector retrieval, compare it honestly with keyword search on labelled held-out data, and tune a hybrid semantic + keyword retriever.

### What is included
- Flexible CSV loader for resumes/JDs and common PlaceMux `candidates.csv` / `jobs.csv` formats.
- Sentence-Transformer embeddings with normalized vector index.
- FAISS vector retrieval when installed.
- BM25-style keyword retrieval using TF-IDF cosine similarity.
- Hybrid retrieval with tunable semantic weight.
- Labelled evaluation with Recall@K, Precision@K, MRR and nDCG@K.
- Train/eval split so hybrid weighting is tuned only on train labels.
- Reproducible experiment log under `logs/`.
- Explainable result reasons from keyword overlap + semantic similarity.
- FastAPI `/search`, `/health`, and `/reload` endpoints.
- Safe lexical fallback when the embedding model/index is unavailable.
- Tests for loader, retrieval, tuning, evaluation and API behavior.

### Required real data
Put these in `data/`:

**Preferred contract**
- `resumes.csv`: `resume_id`, `text` (optional `candidate_id` accepted)
- `jobs.csv`: `job_id`, `text` (or compatible title/description fields)
- `eval_labels.csv`: `query_id`, `query`, `doc_id`, `label`

`label >= 1` means relevant; `label = 0` means non-relevant.

**PlaceMux-compatible input**
The loader also accepts `candidates.csv` and `jobs.csv` and builds resume text from common candidate fields. If an `interactions.csv` exists, it can be used as a source for positive labels, but a dedicated `eval_labels.csv` is recommended for a proper labelled evaluation.

### Run
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the real CSVs into `data/`, then:

```powershell
python run.py --model sentence-transformers/all-MiniLM-L6-v2
python -m pytest -q
uvicorn src.api:app --reload
```

Search:
```powershell
curl "http://127.0.0.1:8000/search?q=someone%20who%20can%20build%20data%20pipelines&top_k=5&mode=hybrid"
```

### Evaluation discipline
- The labelled rows are split by `query_id`, not individual rows, so one query cannot leak across train/eval.
- Hybrid weight is tuned only on the training query groups.
- Final semantic and hybrid metrics are reported on the untouched held-out query groups.
- The project does not claim an offline win unless the held-out metrics demonstrate it.

### Demo flow
`run.py` prints one worked example with query, results, reasons and fallback behavior. The API's `mode=keyword|semantic|hybrid` allows live comparison.
