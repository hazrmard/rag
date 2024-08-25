# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %reload_ext autoreload
# %autoreload 2
import os
import re
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from litellm import embedding, completion, completion_cost
import json

MODEL_NAME = "gpt-4o-mini"

load_dotenv("./.env", override=True);

# %% [markdown]
# ## Download

# %%
url = "https://api.openquran.com"
intro_url = "/express/chapter/intro/{ch}"
verses_url = "/express/chapter/{ch}:{start}-{end}"

payload = {"en":False,"zk":False,"sc":False,"v5":True,"cn":False,"sp_en":False,"sp_ur":False,"ur":False,"ts":False,"fr":False,"es":False,"de":False,"it":False,"my":False,"f":1,"hover":0}

# %%
import requests, json


# %%
def get_ch(n=1, start=1, end=300):
    vdata = []
    for s in range(start, end, 10):
        res=requests.post((url+verses_url).format(ch=n, start=s, end=s+10), json=payload)
        data = json.loads(res.content)
        if data == {'message': 'invalid request'}:
            break
        vdata.extend(data)
    res = requests.get((url+intro_url).format(ch=n))
    intro_data = json.loads(res.content)
    return dict(**intro_data, verses=vdata)


# %%
quran = {}
for i in range(1, 115):
    quran[i] = get_ch(i)

# %%
with open("./quran.json", "w") as f:
    json.dump(quran, f)

# %% [markdown]
# ## Parsing

# %%
with open('./quran.json', 'r') as f:
    data = json.load(f)
quran = data = {int(k): v for k,v in data.items()}

# %%
len(data[5]['verses'])

# %%
data[1]['verses'][0]

# %%
embedding(model='text-embedding-3-small', input=['yo whats up', 'nothing much bro'])


# %%
def sanitize(s: str):
    """Remove [ ] footnote markers from text"""
    return re.sub('<[^<]+?>', '', s)
    


# %%
def get_verses(ch: dict) -> dict[int, str]:
    v = {}
    for d in ch['verses']:
        v[d['v']] = sanitize(' '.join(w['t'] for w in d['words'] if w['t'] is not None))
    return v
get_verses(data[1])


# %%
verses = {i: get_verses(d) for i, d in data.items()}


# %%
def get_topics(q: dict=quran):
    t = set()
    for ch in q.values():
        for v in ch['verses']:
            for topic in v['topics']:
                t.add(topic['topic'])
    return t
t = get_topics(quran)

# %%
sorted(t)


# %%
def get_metadata(ch: dict) -> dict[int, dict[str, str]]:
    m = {}
    for d in ch['verses']:
        m_ = dict(
            ch=d['ch'],
            v=d['v'],
            notes='\n'.join(n['note'] for n in d['v5']['notes'])
        )
        m[d['v']] = m_
    return m

get_metadata(data[1])


# %% [markdown]
# ### DB

# %%
import chromadb
# chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection(name="quran")

# %%
for c in range(1, 115):
    v = get_verses(quran[c])
    m = get_metadata(quran[c])
    
    collection.upsert(
        ids=['%d:%d'%(c,i) for i in v.keys()],
        documents=list(v.values()),
        metadatas=list(m.values())
    )

# %%
r=collection.query(query_texts='actions of the people that disobeyed God', n_results=5)

# %%
r['ids']

# %% [markdown]
# ### LLM

# %%
from framework import system_prompt as prompt


# %%
def find(question, collection, n=10):
    res = collection.query(query_texts=question, n_results=n)
    context = '\n'.join('%s: %s' % (res['ids'][0][i], res['documents'][0][i]) for i in range(len(res['ids'])))
    return context


# %%
def prepare_prompt(question, collection=collection, n=10):
    context = find(question, collection, n=n)
    p = prompt.format(context=context, question=question)
    return p


# %%
p=prepare_prompt('What is the story of moses?')

# %%
print(p)

# %%
res=completion(model=MODEL_NAME, temperature=0, messages=[{"role": "system", "content":p}])

# %%
res['choices'][0]['message'].content


# %%
def router(resp: str, collection=collection, **kwargs):
    kind, val = resp.split(':', maxsplit=1)
    if kind=='ANSWER':
        return val
    elif kind=='FOLLOWUP':
        return val
    elif kind=='UNKNOWN':
        return 'Unable to answer question. Either the lookup failed or the question could not be understood.'
    elif kind=='FIND':
        return '<EXCERPT>\n%s\n</EXCERPT>' % find(val, collection, n=kwargs.get('n', 10))
    elif kind=='CONTEXT':
        verses = val.strip().split(',')
        ctx = []
        for chv in verses:
            ch, v = chv.strip().split(':')
            ids = [f'{ch}:{i}' for i in range(max(int(v)-2, 1), int(v)+3)]
            res = collection.get(ids=ids)
            if len(res):
                context = '\n'.join('%s: %s' % (res['ids'][i], res['documents'][i]) for i in range(len(res['ids'])))
                ctx.append(context)
        return '<EXCERPT>\n' + '\n'.join(ctx) + '\n</EXCERPT>'


# %%
print(router('FIND: Moses'))
