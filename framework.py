import chromadb
from pathlib import Path

collection_name = "quran"
filepath = Path(__file__).parent.resolve()

system_prompt=\
"""\
You are a scholarly assistant, with expertise in objectively analyzing historical and religious texts.
You will be provided a some excerpts from a text. Your job is to use the excerpts, and only the excerpts,
to answer a user's question. The user may ask follow-up questions. You may be given multiple excerpts.

The excerpts will be provided with their chapter and verse numbers, like following:

    [CHAPTER:VERSE] TEXT
    [CHAPTER:VERSE] TEXT

Your response must follow this format:

    RESPONSE_TYPE: RESPONSE_VALUE

You have the following choices:

1. If the excerpts provided do not provide enough context, ask for more context. For example, if the verse ends before
finishing an idea, or if it begins mid-thought. You will be given 2 verses before and after the requested excerpt:

    CONTEXT: CHAPTER:VERSE

You may ask for multuple excerpts:

    CONTEXT: CHAPTER:VERSE, CHAPTER:VERSE

2. If you are able to answer the question:

    ANSWER: YOUR RESPONSE

3. If the question needs more specificity:

    FOLLOWUP: FOLLOW-UP QUESTION TO ASK FOR CLARITY.

4. If you need to look up additional excerpts, you must provide a query string which will be used to find relevant verses. A question is not a query
string. You must turn the question into a search string:

    FIND: QUERY STRING

4. If the question, excerpts, and follow-ups are insufficient to make a response:

    UNKNOWN: 

You may begin with the following excerpt and questions:

<EXCERPT>
{context}
</EXCERPT>

<QUESTION>
{question}
</QUESTION>
"""


def get_collection() -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection


def find(question: str, collection: chromadb.Collection, n=10) -> str:
    res = collection.query(query_texts=question, n_results=n)
    context = '\n'.join('%s: %s' % (res['ids'][0][i], res['documents'][0][i]) for i in range(len(res['ids'])))
    return context


def prepare_prompt(question: str, collection: chromadb.Collection, n=10, prompt=system_prompt):
    context = find(question, collection=collection, n=n)
    p = prompt.format(context=context, question=question)
    return p


def router(resp: str, _collection=chromadb.Collection, **kwargs):
    kind, val = resp.split(':', maxsplit=1)
    if kind=='ANSWER':
        return val
    elif kind=='FOLLOWUP':
        return val
    elif kind=='UNKNOWN':
        return 'Unable to answer question. Either the lookup failed or the question could not be understood.'
    elif kind=='FIND':
        return '<EXCERPT>\n%s\n</EXCERPT>' % find(val, collection=_collection, n=kwargs.get('n', 10))
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
        return '<EXCERPT>\n' + '\n'.join(ctx) + '\n</EXCERPT>'
    else:
        return 'RESPONSE ERROR. The model responded with unknown format: \n%s' % resp
