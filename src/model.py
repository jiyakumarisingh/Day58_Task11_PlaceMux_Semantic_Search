from __future__ import annotations
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def load_embedding_model(name):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    return SentenceTransformer(name)
