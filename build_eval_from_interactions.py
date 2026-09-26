from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA=Path('data')

def pick(df, names):
    for n in names:
        if n in df.columns: return n
    return None

c=pd.read_csv(DATA/'candidates.csv').fillna('')
j=pd.read_csv(DATA/'jobs.csv').fillna('')
i=pd.read_csv(DATA/'interactions.csv').fillna('')

cid=pick(c,['candidate_id','resume_id','id'])
jid=pick(j,['job_id','jd_id','id'])
icid=pick(i,['candidate_id','student_id','resume_id','candidate'])
ijid=pick(i,['job_id','jd_id','job'])
if not all([cid,jid,icid,ijid]):
    raise ValueError(f'Could not detect IDs. candidates={list(c.columns)}, jobs={list(j.columns)}, interactions={list(i.columns)}')

outcome=pick(i,['label','relevance','relevant','hired','applied','selected','clicked','accepted','outcome','score'])
if outcome:
    raw=pd.to_numeric(i[outcome],errors='coerce').fillna(0)
    pos=raw>0
else:
    # If the log has no outcome field, an interaction itself is treated as a positive event.
    # This is a logged-behavior label, not a manually curated relevance judgment.
    pos=pd.Series(True,index=i.index)

# Build query text from all textual job fields.
def text_for(row, id_col):
    vals=[]
    for col,val in row.items():
        if col==id_col: continue
        if isinstance(val,str) and val.strip(): vals.append(val.strip())
    return ' '.join(vals)

jobs={str(r[jid]):text_for(r,jid) for _,r in j.iterrows()}
candidates=[str(x) for x in c[cid].tolist()]
positive=set((str(r[ijid]),str(r[icid])) for _,r in i[pos].iterrows())
rows=[]
for job_id,query in jobs.items():
    if not query.strip(): continue
    positives={cand for job,cand in positive if job==job_id}
    for cand in candidates:
        rows.append({'query_id':f'job_{job_id}','query':query,'doc_id':cand,'label':1 if cand in positives else 0})

out=pd.DataFrame(rows)
out.to_csv(DATA/'eval_labels.csv',index=False)
print(f'Created data/eval_labels.csv: {len(out)} rows, {out.query_id.nunique()} query groups, positives={int(out.label.sum())}')
print('Source: real candidates.csv + jobs.csv + interactions.csv; no manually selected examples were added.')
