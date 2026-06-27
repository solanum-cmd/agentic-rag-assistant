import streamlit as st
from app.chat import ask_llm

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Assistant")

question = st.text_input(
    "Ask anything"
)

if st.button("Send"):

    if question:

        with st.spinner("Thinking..."):

            answer = ask_llm(question)

        st.write(answer)