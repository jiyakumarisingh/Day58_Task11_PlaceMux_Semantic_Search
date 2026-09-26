from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .data import load_documents
from .retrieval import SearchEngine
from .model import load_embedding_model


DATA_DIR = Path(os.getenv("PLACEMUX_DATA_DIR", "data"))
MODEL_NAME = os.getenv(
    "PLACEMUX_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

engine = None
SEM_WEIGHT = 1.0


def build_engine():
    global engine

    resumes, jobs = load_documents(DATA_DIR)

    docs = list(
        zip(
            resumes.doc_id.astype(str),
            resumes._text.astype(str),
        )
    ) + list(
        zip(
            jobs.doc_id.astype(str),
            jobs._text.astype(str),
        )
    )

    try:
        model = load_embedding_model(MODEL_NAME)
    except Exception:
        model = None

    engine = SearchEngine(
        [x[0] for x in docs],
        [x[1] for x in docs],
        model=model,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        build_engine()
    except Exception:
        pass
    yield


app = FastAPI(
    title="PlaceMux Day 58 Semantic Search",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "semantic_available": bool(
            engine and engine.model is not None
        ),
        "semantic_weight": SEM_WEIGHT,
    }


@app.post("/reload")
def reload_index():
    try:
        build_engine()
        return {
            "status": "reloaded",
            "semantic_available": bool(
                engine and engine.model is not None
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
def search(
    q: str,
    top_k: int = 5,
    mode: str = "hybrid",
    semantic_weight: float | None = None,
):
    if not q.strip():
        raise HTTPException(
            status_code=400,
            detail="q cannot be empty",
        )

    if engine is None:
        try:
            build_engine()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Search index unavailable: {e}",
            )

    if mode not in {"keyword", "semantic", "hybrid"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be keyword, semantic or hybrid",
        )

    return engine.search(
        q,
        mode=mode,
        top_k=max(1, min(top_k, 50)),
        semantic_weight=(
            SEM_WEIGHT
            if semantic_weight is None
            else semantic_weight
        ),
    )