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
from tqdm.autonotebook import trange, tqdm

MODEL_NAME = "gpt-4o-mini"

load_dotenv("./.env", override=True);

# %% [markdown] jp-MarkdownHeadingCollapsed=true
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
quran = {int(k): v for k,v in data.items()}


# %%
# Processing functions for text elements
def sanitize_verse(s: str):
    """Remove [ ] footnote markers and <> tags from text"""
    square_brackets_pattern = r'\[[^\]]*\]'  # Matches anything inside square brackets
    angle_brackets_pattern = r'<[^>]*>'      # Matches anything inside angle brackets
    # Remove content inside square brackets
    result = re.sub(square_brackets_pattern, '', s)
    # Remove content inside angle brackets
    result = re.sub(angle_brackets_pattern, '', result)
    # Remove any extra spaces that may have been introduced
    result = re.sub(r'\s+', ' ', result).strip()
    return result
def sanitize_topic(s: str):
    return sanitize_verse(','.join(s.split(':')))


# %%
def get_verses(ch: dict) -> dict[int, str]:
    v = {}
    for d in ch['verses']:
        # v[d['v']] = sanitize_verse(' '.join(w['t'] for w in d['words'] if w['t'] is not None))
        v[d['v']] = sanitize_verse(d['v5']['text'])
    return v


# %%
def get_topics(q: dict=quran):
    from collections import defaultdict
    reverse = defaultdict(list, {})
    t = set()
    for ch in q.values():
        for v in ch['verses']:
            for topic in v['topics']:
                t_ = sanitize_topic(topic['topic'])
                t.add(t_)
                reverse[t_].append('%d:%d'%(v['ch'], v['v']))
    return t, reverse
# t, reverset = get_topics(quran)


# %%
verses = {i: get_verses(d) for i, d in quran.items()}

# %%
quran[1]['verses'][0]

# %%
verses[1]


# %%
def get_metadata(ch: dict) -> dict[int, dict[str, str]]:
    m = {}
    for d in ch['verses']:
        m_ = dict(
            ch=d['ch'],
            v=d['v'],
            topics='\n'.join(sanitize_topic(t['topic']) for t in d['topics']),
            notes='\n'.join(n['note'] for n in d['v5']['notes'])
        )
        m[d['v']] = m_
    return m


# %%
get_metadata(quran[1])

# %% [markdown]
# ### DB

# %% [markdown]
# #### Verses

# %%
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection(name="quran")

# %%
embed_fn = DefaultEmbeddingFunction()
for c in trange(1, 115, leave=False):
    ch = get_verses(quran[c])
    meta = get_metadata(quran[c])

    embed_strs = []
    for (i, v), (j, m) in zip(ch.items(), meta.items()):
        assert i==j
        embed_str = f"{v}\n\nTOPICS:\n\n{m['topics']}"
        embed_strs.append(embed_str)
    embeddings = embed_fn(embed_strs)

    collection.upsert(
        ids=['%d:%d'%(c,i) for i in ch.keys()],
        embeddings=embeddings,
        documents=list(ch.values()),
        metadatas=list(meta.values())
    )

# %%
c = [[1,2],[3,4,5]]
[a for b in c for a in b]

# %%
r=collection.query(query_texts='actions of the people that disobeyed God', n_results=5)

# %%
r['metadatas'][0][0]

# %%
collection.get('53:53')

# %% [markdown]
# #### Topics

# %%
import chromadb
chroma_client = chromadb.PersistentClient()
topics_collection = chroma_client.get_or_create_collection(name="quran_topics")


# %%
# Prompt --vectorDB--> topic --table lookup--> verses
# Prompt --vectorDB--> verses

# %%
def topic_metadata_to_database(topics, reverse_topics, collection):
    for t in tqdm(topics, leave=False):
        t_ = sanitize_topic(t)
        embed_str = f'What does the Quran say on the topic of: "{t}"'
        collection.upsert(
            ids=[t],
            documents=[embed_str],
            metadatas={'verse_ids':','.join(reverse_topics[t])}
        )
topics, reverse_topics = get_topics(quran)
topic_metadata_to_database(t, reverse_topics, topics_collection)

# %% [markdown]
# ### LLM

# %%
from framework import system_prompt as prompt, router, find, get_collection

# %%
res=completion(model=MODEL_NAME, temperature=0, messages=[{"role": "system", "content":p}])

# %%
res['choices'][0]['message'].content

# %%
print(router('FIND: Moses'))
