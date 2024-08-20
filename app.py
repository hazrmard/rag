import os

import streamlit as st
from dotenv import load_dotenv
from litellm import completion, embedding

load_dotenv("./env", override=True)

st.set_page_config(page_title="RAG", page_icon="📖")

testing = st.toggle('Testing?', value=True)

inp = st.text_area(label='Input')

messages = [
    dict(
        role="system",
        content="You are a medieval bard with knowledge of the future. You respond to questions in the iambic pentameter."
    )
]

if inp:
    try:
        msg = {"role": "user", "content": inp}
        response = completion(
            model="gpt-4o-mini",
            messages=messages+[msg],
            mock_response="Testing" if testing else None
        )
        st.write(response.choices[0].message.content)
    except Exception as exc:
        st.error(exc)
