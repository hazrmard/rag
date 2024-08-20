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

load_dotenv("./.env", override=True);

# %% [markdown]
# ## Download

# %%
url = "https://api.openquran.com"
intro = "/express/chapter/intro/{ch}"
verses = "/express/chapter/{ch}:{start}-{end}"

payload = {"en":False,"zk":False,"sc":False,"v5":True,"cn":False,"sp_en":False,"sp_ur":False,"ur":False,"ts":False,"fr":False,"es":False,"de":False,"it":False,"my":False,"f":1,"hover":0}

# %%
import requests, json


# %%
def get_ch(n=1):
    res=requests.post((url+verses).format(ch=n, start=1, end=1000), json=payload)
    data = json.loads(res.content)
    res = requests.get((url+intro).format(ch=n))
    intro_data = json.loads(res.content)
    return dict(**intro_data, verses=data)


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
data = {int(k): v for k,v in data.items()}

# %%
data[2]['verses'][:1]

# %%
embedding(model='text-embedding-3-small', input=['yo whats up', 'nothing much bro'])


# %%
def sanitize(s: str):
    return re.sub('<[^<]+?>', '', s)
    


# %%
def get_verses(ch: dict):
    v = {}
    for d in ch['verses']:
        v[d['v']] = sanitize(' '.join(w['t'] for w in d['words'] if w['t'] is not None))
    return v
get_verses(data[1])
        
