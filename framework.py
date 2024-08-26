from pathlib import Path
import sys
import re
import chromadb

collection_name = "quran"
filepath = Path(__file__).parent.resolve()
max_loops = 5

system_prompt=\
f"""\
You are a scholarly assistant, with expertise in objectively analyzing historical and religious texts.
You are to look up excerpts of verses from the Quran. Your job is to use the excerpts, and only the excerpts,
to answer a user's question. The user may ask follow-up questions. You may be given multiple excerpts.

The excerpts will be provided with their chapter and verse numbers, like following. Ignore the angle brackets.
They signify placeholders:

    <CHAPTER>:<VERSE> <TEXT>
    <CHAPTER>:<VERSE> <TEXT>

Your response must follow this format:

    <RESPONSE_TYPE>: <RESPONSE_VALUE>

You have the following choices of actions:

1. Look up excerpts. You can look up excerpts multiple times. You must provide a query string 
which will be used to find relevant verses. This is usually the first step. You may use excerpts 
to help formulate following lookup queries for more excerpts. The user's question is not a 
query string. You must turn the question into a query string:

    FIND: <QUERY STRING>

You may ask for multiple queries:
    
    FIND: <QUERY STRING>, <QUERY STRING>

2. Obtain more context by extracting verses before and after the queried excerpts. For example,
if a verse in the excerpt ends before finishing an idea, or if it begins mid-thought. You will be given 2 
verses before and after the verse in question:

    CONTEXT: <CHAPTER>:<VERSE>

You may ask for multuple excerpts:

    CONTEXT: <CHAPTER:VERSE>, <CHAPTER:VERSE>

3. If the question needs to be clarified, ask the user a follow-up:

    FOLLOWUP: <FOLLOW-UP QUESTION TO ASK FOR CLARITY>

4. Make a final answer. If you are able to answer the question, or if repeated FIND, CONTEXT, 
and FOLLOWUP actions were insufficient to make an objective answer. The answer must only be 
based on excerpts, which you must cite as <CHAPTER>:<VERSE>. Do not quote verses. You must not 
make any additional inferences.

    ANSWER: <YOUR RESPONSE AND EXPLANATION>

For example, an analysis could comprise of the following actions:

    # Get excerpts
    FIND: <INITIAL SEARCH QUERY>
    # Get surrounding verses of a particular verse
    CONTEXT: <CHAPTER>:<VERSE>
    # Make additional search queries
    FIND: <A NEW SEARCH QUERY INFORMED BY RESULTS OF THE LAST QUERY>
    # Produce a final objective answer:
    ANSWER: <ANSWER, OR EXPLANATION OF WHY AN ANSWER IS NOT POSSIBLE>

Important: You are allowed at most {max_loops} actions. The last action must be ANSWER.

You may begin with the following question:
"""


def get_collection() -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection


def find(question: str, collection: chromadb.Collection, n=10) -> str:
    res = collection.query(query_texts=question, n_results=n)
    context = '\n\n'.join('%s: %s' % (res['ids'][0][i], res['documents'][0][i]) for i in range(len(res['ids'])))
    return context


def _process_answer(ans: str, collection: chromadb.Collection) -> str:
    pattern = r'(\d+:\d+)'
    matches = re.findall(pattern=pattern, string=ans)
    vdata = collection.get(ids=list(set(matches)))
    res = '\n\n'.join('[%s]: %s' % (loc, verse) for loc, verse in zip(matches, vdata['documents']))
    return ans + '\n\nReferences:\n\n' + res


def router(resp: str, _collection=chromadb.Collection, **kwargs) -> tuple[str, bool, bool]: # ans, back_to_llm, display
    kind, val = resp.split(':', maxsplit=1)
    if kind=='ANSWER':
        return _process_answer(val, collection=_collection), False, True
    elif kind=='FOLLOWUP':
        return val, False, True
    elif kind=='FIND':
        res = []
        queries = val.split(',')
        for q in queries:
            res.append(find(q, collection=_collection, n=kwargs.get('n', 10)))
        return '<EXCERPT>\n\n%s\n\n</EXCERPT>' % '\n\n'.join(res), True, False
    elif kind=='CONTEXT':
        verses = val.strip().split(',')
        ctx = []
        for chv in verses:
            ch, v = chv.strip().split(':')
            ids = [f'{ch}:{i}' for i in range(max(int(v)-2, 1), int(v)+3)]
            res = _collection.get(ids=ids)
            if len(res):
                context = '\n'.join('%s: %s' % (res['ids'][i], res['documents'][i]) for i in range(len(res['ids'])))
                ctx.append(context)
        return '<EXCERPT>\n\n' + '\n\n'.join(ctx) + '\n\n</EXCERPT>', True, False
    else:
        return 'RESPONSE ERROR. The model responded with unknown format: \n%s' % resp, True, False
