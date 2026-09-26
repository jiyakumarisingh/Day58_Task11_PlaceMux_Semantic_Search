# PlaceMux — Day 58 / Task 11

## Semantic Search, Vector Retrieval & Hybrid Search

### Objective

Ship semantic search over resumes and job descriptions using embeddings and vector retrieval, compare it honestly with keyword search on labelled held-out data, and tune a hybrid semantic + keyword retriever.

---

## What Is Included

* Flexible CSV loader for resumes/JDs and common PlaceMux `candidates.csv` / `jobs.csv` formats.
* Sentence-Transformer embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
* Normalized vector representations for semantic retrieval.
* FAISS vector retrieval when FAISS is available.
* BM25-style keyword retrieval using TF-IDF cosine similarity.
* Hybrid retrieval with tunable semantic weighting.
* Labelled evaluation using:

  * Precision@K
  * Recall@K
  * MRR
  * nDCG@K
* Query-level train/evaluation split to prevent query leakage.
* Hybrid semantic-weight tuning using training query groups only.
* Held-out evaluation on untouched query groups.
* Reproducible experiment log under `logs/`.
* Explainable search results using semantic similarity and keyword overlap.
* FastAPI serving endpoints:

  * `/health`
  * `/search`
  * `/reload`
* Semantic search with automatic keyword fallback when the embedding model is unavailable.
* Automated tests covering data loading, retrieval, evaluation, tuning, and API behavior.

---

## Project Structure

```text
Day58_Task11_PlaceMux_Semantic_Search/
│
├── data/
│   ├── candidates.csv
│   ├── jobs.csv
│   ├── interactions.csv
│   └── eval_labels.csv
│
├── logs/
│   └── day58_experiment.json
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data.py
│   ├── evaluation.py
│   ├── model.py
│   └── retrieval.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_evaluation.py
│   └── test_retrieval.py
│
├── build_eval_from_interactions.py
├── run.py
├── requirements.txt
├── pytest.ini
└── README.md
```

---

# Data Contract

The system supports both the preferred generic contract and PlaceMux-compatible source data.

## Preferred Input

### `resumes.csv`

Expected fields:

```text
resume_id,text
```

An optional `candidate_id` can also be supplied.

### `jobs.csv`

Expected fields:

```text
job_id,text
```

Compatible title/description fields can also be used.

### `eval_labels.csv`

Expected fields:

```text
query_id,query,doc_id,label
```

Label semantics:

```text
label >= 1  → relevant
label = 0   → non-relevant
```

---

# PlaceMux-Compatible Input

The loader also supports the real PlaceMux format:

### `candidates.csv`

Typical fields:

```text
candidate_id
name
verified_skills
experience_years
education
availability
```

Candidate information is converted into searchable resume-style text.

### `jobs.csv`

Typical fields:

```text
job_id
title
company
required_skills
min_experience_years
category
```

Job information is converted into searchable job-description text.

### `interactions.csv`

Typical fields:

```text
student_id
job_id
relevant
```

When available, interactions can be used to build evaluation labels.

The evaluation builder supports both `candidate_id` and PlaceMux's `student_id` identifier.

---

# Installation

Create the virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# Build Evaluation Labels

When using the PlaceMux interaction data:

```powershell
python build_eval_from_interactions.py
```

This generates:

```text
data/eval_labels.csv
```

The evaluation data is generated from the real candidate, job, and interaction datasets rather than manually selecting demonstration examples.

---

# Run Offline Evaluation

Run the complete experiment:

```powershell
python run.py
```

Or specify the embedding model:

```powershell
python run.py --model sentence-transformers/all-MiniLM-L6-v2
```

The experiment performs:

1. Real data loading.
2. Embedding model loading.
3. Vector indexing.
4. Keyword retrieval.
5. Semantic retrieval.
6. Hybrid retrieval.
7. Query-level train/test splitting.
8. Hybrid weight tuning.
9. Held-out evaluation.
10. Worked-example retrieval.
11. Fallback validation.
12. Experiment logging.

---

# Evaluation Methodology

The evaluation is performed at the **query level**.

This is important because individual labelled rows from the same query must not be split between training and evaluation.

The pipeline therefore:

1. Groups labels by `query_id`.
2. Splits query groups into training and held-out evaluation groups.
3. Tunes the semantic weight using training queries only.
4. Keeps held-out queries untouched.
5. Reports final metrics on the held-out set.

This avoids query-level leakage.

---

# Metrics

The project reports:

### Precision@K

Measures the proportion of retrieved documents that are relevant.

### Recall@K

Measures how many of the relevant documents are retrieved within the top K results.

### MRR

Mean Reciprocal Rank measures how early the first relevant result appears.

### nDCG@K

Normalized Discounted Cumulative Gain measures ranking quality while giving more importance to highly ranked relevant documents.

---

# Retrieval Modes

The API supports three retrieval modes.

## Keyword

Uses lexical similarity:

```text
mode=keyword
```

Useful when exact terms and skills matter.

## Semantic

Uses embedding similarity:

```text
mode=semantic
```

Useful when the query and candidate/JD use related concepts rather than identical words.

## Hybrid

Combines semantic and keyword scores:

```text
mode=hybrid
```

The semantic contribution is controlled by `semantic_weight`.

---

# Hybrid Weight Tuning

The semantic weight was evaluated from:

```text
0.0
0.1
0.2
...
1.0
```

The final experiment selected:

```text
semantic_weight = 1.0
```

based on the training-query evaluation.

The experiment results are stored in:

```text
logs/day58_experiment.json
```

---

# Evaluation Results

Held-out evaluation:

| Method   | Precision@5 | Recall@5 |    MRR | nDCG@5 |
| -------- | ----------: | -------: | -----: | -----: |
| Keyword  |      0.4000 |   0.8889 | 0.5000 | 0.6392 |
| Semantic |      0.4000 |   0.8889 | 1.0000 | 0.9218 |
| Hybrid   |      0.4000 |   0.8889 | 1.0000 | 0.9218 |

Held-out evaluation contains:

```text
Test query groups: 3
```

The semantic retriever improves ranking quality over the keyword baseline on this held-out set, particularly:

```text
MRR:
Keyword   = 0.5000
Semantic  = 1.0000

nDCG@5:
Keyword   = 0.6392
Semantic  = 0.9218
```

The hybrid result is equal to semantic retrieval on this small held-out evaluation set.

Because the held-out set contains only three query groups, these results should be interpreted as an offline demonstration rather than a production-scale performance guarantee.

---

# Training Weight Tuning

The training-query tuning produced the following trend:

| Semantic Weight | Precision@5 | Recall@5 |    MRR | nDCG@5 |
| --------------: | ----------: | -------: | -----: | -----: |
|             0.0 |      0.4286 |   0.7500 | 0.4286 | 0.5227 |
|             0.3 |      0.4857 |   0.8929 | 0.4762 | 0.6138 |
|             0.5 |      0.5143 |   0.9286 | 0.5476 | 0.6840 |
|             0.7 |      0.5429 |   0.9643 | 0.7857 | 0.8149 |
|             0.8 |      0.5429 |   0.9643 | 0.8571 | 0.8705 |
|             0.9 |      0.5429 |   0.9643 | 0.8571 | 0.8744 |
|             1.0 |      0.5429 |   0.9643 | 0.9286 | 0.9023 |

The selected semantic weight is:

```text
1.0
```

---

# Worked Search Example

Example recruiter query:

```text
someone who can build data pipelines
```

The semantic retrieval path identifies documents based on meaning rather than requiring exact keyword matches.

Example semantic results include:

```text
101 — Junior Data Scientist
110 — Data Science Intern
102 — Data Analyst
107 — Cloud Developer
108 — NLP Data Scientist
```

Notably, candidate/document `107` received:

```text
semantic_score = 0.489393
keyword_score  = 0.0
```

This demonstrates that semantic retrieval can surface a document even when there is no exact query-term overlap.

---

# Explainability

Each result contains:

```text
score
semantic_score
keyword_score
reason
text
```

Example reason:

```text
Semantic similarity plus matching terms: data.
```

When there is no exact keyword overlap but semantic similarity is high, the system explains:

```text
High semantic similarity even without exact query-term overlap.
```

---

# FastAPI

Start the API:

```powershell
python -m uvicorn src.api:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

---

# Health Check

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/health" | ConvertTo-Json
```

Expected response includes:

```json
{
  "status": "ok",
  "semantic_available": true,
  "semantic_weight": 1.0
}
```

---

# Semantic Search

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/search?q=someone%20who%20can%20build%20data%20pipelines&top_k=5&mode=semantic" | ConvertTo-Json -Depth 10
```

---

# Keyword Search

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/search?q=someone%20who%20can%20build%20data%20pipelines&top_k=5&mode=keyword" | ConvertTo-Json -Depth 10
```

---

# Hybrid Search

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/search?q=someone%20who%20can%20build%20data%20pipelines&top_k=5&mode=hybrid" | ConvertTo-Json -Depth 10
```

---

# Reload Index

```powershell
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/reload" | ConvertTo-Json
```

Expected:

```json
{
  "status": "reloaded",
  "semantic_available": true
}
```

---

# Safe Fallback

If the embedding model becomes unavailable, the API does not fail the search request.

Instead:

```text
semantic request
       ↓
embedding unavailable
       ↓
keyword fallback
       ↓
results returned
```

The fallback was tested by forcing an invalid embedding model configuration.

The API reported:

```json
{
  "status": "ok",
  "semantic_available": false
}
```

A semantic search request then returned:

```text
mode = keyword_fallback
```

with valid keyword results.

This provides a safe degradation path instead of making the search API unavailable.

---

# Automated Tests

Run:

```powershell
python -m pytest -q
```

Current result:

```text
4 passed
```

The remaining Starlette/httpx message is a dependency-level deprecation warning from the testing stack and does not indicate a failing test.

---

# API Contract

## `GET /health`

Returns service status and semantic availability.

## `GET /search`

Parameters:

```text
q
top_k
mode
semantic_weight
```

Supported modes:

```text
keyword
semantic
hybrid
```

## `POST /reload`

Reloads the document index and embedding model.

---

# Reproducibility

The project records the experiment configuration and evaluation results in:

```text
logs/day58_experiment.json
```

The experiment records:

* embedding model
* K value
* train query count
* test query count
* tuned semantic weight
* keyword metrics
* semantic metrics
* hybrid metrics
* semantic-weight tuning results
* worked-example query

---

# Safety and Failure Handling

The search service includes:

* model loading failure handling
* semantic availability detection
* keyword fallback
* bounded `top_k`
* empty-query validation
* unsupported-mode validation
* API error responses
* reload support

`top_k` is limited to:

```text
1–50
```

---

# Task Completion Checklist

* [x] Real PlaceMux candidate and job data loaded.
* [x] Embedding model integrated.
* [x] Vector retrieval implemented.
* [x] Keyword retrieval implemented.
* [x] Hybrid retrieval implemented.
* [x] Labelled evaluation created from real interactions.
* [x] Query-level train/held-out split implemented.
* [x] Hybrid weight tuned only on training queries.
* [x] Held-out semantic evaluation completed.
* [x] Semantic ranking improvement demonstrated.
* [x] Explainable search reasons implemented.
* [x] FastAPI search endpoint implemented.
* [x] Health endpoint implemented.
* [x] Reload endpoint implemented.
* [x] Keyword fallback implemented.
* [x] Forced semantic failure path tested.
* [x] Automated tests passing.
* [x] Reproducible experiment log generated.

---

# Limitations

This implementation is an offline evaluation and serving prototype.

The current labelled evaluation contains only a small number of held-out query groups. Therefore, the measured improvement should not be interpreted as a production-scale guarantee.

The next validation step for production would be a larger labelled evaluation set followed by online measurement such as search engagement, recruiter actions, candidate views, applications, or downstream matching outcomes.

---

# Hand-off

The search API is ready to be consumed by the PlaceMux backend or frontend through:

```text
GET /search
GET /health
POST /reload
```

The backend/frontend can select:

```text
keyword
semantic
hybrid
```

depending on the desired retrieval behavior.

---

## Task

**PlaceMux — Day 58 / Task 11**

**Semantic Search, Vector Retrieval & Hybrid Search**

Status:

```text
IMPLEMENTED
```
