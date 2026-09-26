from __future__ import annotations
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss
except Exception:
    faiss = None


def tokenize(text):
    return re.findall(r"[a-z0-9+#.]+", str(text).lower())


class SearchEngine:
    def __init__(self, doc_ids, texts, model=None):
        self.doc_ids = list(map(str, doc_ids))
        self.texts = list(map(str, texts))
        self.model = model
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.kw_matrix = self.tfidf.fit_transform(self.texts)
        self.embeddings = None
        self.faiss_index = None
        if model is not None:
            self.embeddings = np.asarray(model.encode(self.texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")
            if faiss is not None:
                self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.faiss_index.add(self.embeddings)

    def _keyword(self, query, k):
        q = self.tfidf.transform([query])
        scores = cosine_similarity(q, self.kw_matrix).ravel()
        order = np.argsort(-scores)[:k]
        return {int(i): float(scores[i]) for i in order}

    def _semantic(self, query, k):
        if self.model is None or self.embeddings is None:
            return {}
        q = np.asarray(self.model.encode([query], normalize_embeddings=True, show_progress_bar=False), dtype="float32")
        if self.faiss_index is not None:
            scores, idx = self.faiss_index.search(q, min(k, len(self.doc_ids)))
            return {int(i): float(s) for i, s in zip(idx[0], scores[0]) if i >= 0}
        scores = (self.embeddings @ q[0]).ravel()
        order = np.argsort(-scores)[:k]
        return {int(i): float(scores[i]) for i in order}

    def search(self, query, mode="hybrid", top_k=10, semantic_weight=0.7):
        kw = self._keyword(query, max(top_k * 4, top_k))
        sem = self._semantic(query, max(top_k * 4, top_k))
        if mode == "keyword":
            combined = {i: s for i, s in kw.items()}
        elif mode == "semantic":
            if not sem:
                combined = {i: s for i, s in kw.items()}
                mode = "keyword_fallback"
            else:
                combined = {i: s for i, s in sem.items()}
        else:
            ids = set(kw) | set(sem)
            combined = {}
            for i in ids:
                ks = kw.get(i, 0.0)
                ss = sem.get(i, 0.0)
                combined[i] = semantic_weight * ss + (1 - semantic_weight) * ks
        order = sorted(combined, key=combined.get, reverse=True)[:top_k]
        results = []
        q_tokens = set(tokenize(query))
        for i in order:
            d_tokens = set(tokenize(self.texts[i]))
            overlap = sorted(q_tokens & d_tokens)
            results.append({
                "doc_id": self.doc_ids[i],
                "score": round(float(combined[i]), 6),
                "semantic_score": round(float(sem.get(i, 0.0)), 6),
                "keyword_score": round(float(kw.get(i, 0.0)), 6),
                "reason": self._reason(overlap, sem.get(i, 0.0), kw.get(i, 0.0)),
                "text": self.texts[i],
            })
        return {"mode": mode, "query": query, "semantic_weight": semantic_weight, "results": results}

    @staticmethod
    def _reason(overlap, semantic, keyword):
        if overlap:
            return f"Semantic similarity plus matching terms: {', '.join(overlap[:6])}."
        if semantic > 0:
            return "High semantic similarity even without exact query-term overlap."
        return "Matched using lexical similarity fallback."
