from app.text_splitter import split_text
from app.vector_store import get_vectorstore
from app.retriever import retrieve

import streamlit as st
from app.chat import ask_llm
from app.pdf_reader import read_pdf

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Assistant")

st.sidebar.title("Upload Documents")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

pdf_text = ""

if uploaded_file:

    pdf_text = read_pdf(uploaded_file)
    chunks = split_text(pdf_text)
    db = get_vectorstore(chunks)

    st.sidebar.success("PDF Loaded Successfully")

question = st.text_input("Ask anything")

if st.button("Send"):

    if question:

        if pdf_text != "":

            context = retrieve(db, question)

            prompt = f"""
Answer ONLY using the following context.

If the answer is not present,
say you don't know.

Context:

{context}

Question:

{question}
"""

        else:

            prompt = question

        with st.spinner("Thinking..."):

            answer = ask_llm(prompt)

        st.write(answer)
        