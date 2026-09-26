from __future__ import annotations
from pathlib import Path
import pandas as pd


def _first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def _text_columns(df, exclude=()):
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            cols.append(c)
    return cols


def load_documents(data_dir: Path):
    data_dir = Path(data_dir)
    resume_file = data_dir / "resumes.csv"
    candidate_file = data_dir / "candidates.csv"
    job_file = data_dir / "jobs.csv"

    if resume_file.exists():
        r = pd.read_csv(resume_file).fillna("")
        rid = _first_existing(r, ["resume_id", "candidate_id", "id"])
        txt = _first_existing(r, ["text", "resume_text", "content", "profile", "skills", "description"])
        if not rid:
            raise ValueError("resumes.csv needs resume_id/candidate_id/id")
        if not txt:
            cols = _text_columns(r, [rid])
            r["_text"] = r[cols].astype(str).agg(" ".join, axis=1) if cols else ""
        else:
            r["_text"] = r[txt].astype(str)
        resumes = r[[rid, "_text"]].rename(columns={rid: "doc_id"})
    elif candidate_file.exists():
        r = pd.read_csv(candidate_file).fillna("")
        rid = _first_existing(r, ["candidate_id", "resume_id", "id"])
        if not rid:
            raise ValueError("candidates.csv needs candidate_id/resume_id/id")
        cols = _text_columns(r, [rid])
        r["_text"] = r[cols].astype(str).agg(" ".join, axis=1) if cols else ""
        resumes = r[[rid, "_text"]].rename(columns={rid: "doc_id"})
    else:
        raise FileNotFoundError("Add data/resumes.csv or data/candidates.csv")

    if not job_file.exists():
        raise FileNotFoundError("Add data/jobs.csv")
    j = pd.read_csv(job_file).fillna("")
    jid = _first_existing(j, ["job_id", "jd_id", "id"])
    txt = _first_existing(j, ["text", "job_text", "description", "job_description", "title"])
    if not jid:
        raise ValueError("jobs.csv needs job_id/jd_id/id")
    if txt:
        j["_text"] = j[txt].astype(str)
    else:
        cols = _text_columns(j, [jid])
        j["_text"] = j[cols].astype(str).agg(" ".join, axis=1) if cols else ""
    jobs = j[[jid, "_text"]].rename(columns={jid: "doc_id"})
    return resumes.drop_duplicates("doc_id"), jobs.drop_duplicates("doc_id")


def load_labels(data_dir: Path):
    p = Path(data_dir) / "eval_labels.csv"
    if not p.exists():
        raise FileNotFoundError("Add data/eval_labels.csv with query_id,query,doc_id,label")
    df = pd.read_csv(p).fillna("")
    required = {"query_id", "query", "doc_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"eval_labels.csv missing columns: {sorted(missing)}")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df
