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

    st.sidebar.success("PDF Loaded Successfully")

question = st.text_input("Ask anything")

if st.button("Send"):

    if question:

        if pdf_text != "":

            prompt = f"""
Use the following document.

Document:

{pdf_text}

Question:

{question}
"""

        else:

            prompt = question

        with st.spinner("Thinking..."):

            answer = ask_llm(prompt)

        st.write(answer)