import os

import streamlit as st
from dotenv import load_dotenv
from litellm import completion

from framework import get_collection, router, prepare_prompt

load_dotenv("./env", override=True)

get_collection = st.cache_resource(get_collection)
router = st.cache_data(router)

st.set_page_config(page_title="RAG", page_icon="📖")

testing = st.toggle('Testing?', value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

inp = st.chat_input()

if inp:
    try:
        if len(st.session_state.messages) == 0:
            st.session_state.messages.append(
                {"role": "system", "content": prepare_prompt(question=inp, collection=get_collection())}
            )
        msg = {"role": "user", "content": inp}
        st.session_state.messages.append(msg)
        response = completion(
            model="gpt-4o-mini",
            messages=st.session_state.messages,
            mock_response="Testing" if testing else None
        )
        response = response.choices[0].message.content
        response = {"role": "assistant", "content": "Testing" if testing else router(response, _collection=get_collection())}
        st.session_state.messages.append(response)
    except Exception as exc:
        st.error(exc)

for msg in st.session_state.messages:
    with st.chat_message(name=msg["role"]):
        st.write(msg["content"])
