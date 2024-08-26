import os

import streamlit as st
from dotenv import load_dotenv
from litellm import completion

from framework import get_collection, router, system_prompt, max_loops

load_dotenv("./env", override=True)

get_collection = st.cache_resource(get_collection)

st.set_page_config(page_title="Q-bot", page_icon="📖")
st.caption('AI-powered Quran Assistant. Ask general questions & follow-ups, refer to verses etc.')

testing = st.toggle('Testing?', value=False)
intermediate = st.toggle('Show intermediate answers?', value=False)

def add_message(role: str, content: str, display: bool=True, store: bool=True):
    st.session_state.messages.append({"role": role, "content": content})
    st.session_state.display.append(display)
    st.session_state.store.append(store)
    assert len(st.session_state.messages) == len(st.session_state.display)
    assert len(st.session_state.store) == len(st.session_state.display)

if "messages" not in st.session_state:
    st.session_state.messages = [] # the message dicts
    st.session_state.display = [] # is message to be displayed?
    st.session_state.store = [] # is message stored to be sent to LLM?

inp = st.chat_input('Ask a question from the Quran.')

if inp:
    try:
        if len(st.session_state.messages) == 0:
            add_message("system", system_prompt, False, True)

        add_message("user", inp, display=True, store=True)

        back_to_llm = True
        i = 0
        while back_to_llm and i < max_loops:
            i += 1

            response = completion(
                model="gpt-4o-mini",
                messages=[m for i, m in enumerate(st.session_state.messages) if st.session_state.store[i]],
                mock_response="TESTING: Testing" if testing else None
            )
            response_str = response.choices[0].message.content
            add_message("assistant", response_str, display=intermediate, store=True)

            result_str, back_to_llm, display_msg = router(response_str, _collection=get_collection())

            add_message("assistant", result_str, display=display_msg or intermediate, store=back_to_llm)

    except Exception as exc:
        st.error(exc)
        raise exc

for msg, display in zip(st.session_state.messages[1:], st.session_state.display[1:]):
    if not (display or intermediate):
        continue
    with st.chat_message(name=msg["role"]):
        st.write(msg["content"])
