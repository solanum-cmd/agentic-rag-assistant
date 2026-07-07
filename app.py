"""
Streamlit Web UI for the Agentic RAG Assistant.
"""

import logging
import streamlit as st

from app.agent import AgentRAG
from app.config import validate_config, LLM_PROVIDER, EMBEDDING_PROVIDER

# ---------------------------------------------------------------------------
# Setup & Initialization
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Agent in Session State (singleton for the session)
if "agent" not in st.session_state:
    st.session_state.agent = AgentRAG()

agent: AgentRAG = st.session_state.agent

# Check Configuration Warnings
config_warnings = validate_config()
if config_warnings:
    for warning in config_warnings:
        st.error(f"Configuration Error: {warning}")
    st.stop()


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------
def render_message(role: str, content: str) -> None:
    """Render a chat message in the UI."""
    with st.chat_message(role):
        st.markdown(content)


# ---------------------------------------------------------------------------
# Sidebar: Document Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 DocMind AI")
    st.caption("Enterprise RAG Assistant")
    
    st.divider()
    
    st.header("Document Knowledge Base")
    
    # Status Indicator
    if agent.is_ready:
        count = agent.get_document_count()
        st.success(f"Database Active: {count} chunks loaded")
    else:
        st.warning("Database Empty: Upload PDFs below")

    # Upload form
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to add to the knowledge base."
        )
        submitted = st.form_submit_button("Process Documents", use_container_width=True)
        
        if submitted:
            if uploaded_files:
                with st.spinner("Extracting text and building vector store..."):
                    try:
                        chunks_created = agent.ingest_pdfs(uploaded_files)
                        st.session_state.upload_success = f"Successfully processed {len(uploaded_files)} files ({chunks_created} chunks)."
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please select at least one file.")

    # Show success message from previous run if exists
    if "upload_success" in st.session_state:
        st.success(st.session_state.upload_success)
        del st.session_state.upload_success

    st.divider()

    # Management Actions
    st.header("Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            agent.clear_memory()
            st.rerun()
            
    with col2:
        if st.button("Reset DB", type="primary", use_container_width=True):
            agent.reset_database()
            st.rerun()
            
    st.divider()
    
    with st.expander("System Info"):
        st.markdown(f"**LLM:** `{LLM_PROVIDER}`")
        st.markdown(f"**Embeddings:** `{EMBEDDING_PROVIDER}`")


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

# Render chat history
for msg in agent.memory.get_messages():
    role = "user" if msg.type == "human" else "assistant"
    render_message(role, str(msg.content))

# Chat Input
if question := st.chat_input("Ask a question about your documents..."):
    
    if not agent.is_ready:
        st.warning("Please upload and process at least one PDF first.")
    else:
        # 1. Render user question immediately
        render_message("user", question)
        
        # 2. Process via Agent
        with st.chat_message("assistant"):
            try:
                # We get a streaming generator back from the agent
                result = agent.ask(question)
                answer_stream = result["answer"]
                retrieval_result = result["retrieval_result"]
                
                # Stream the output directly to the UI
                full_response = st.write_stream(answer_stream)
                
                # Update memory with the final complete response
                agent.save_ai_response(full_response)
                
                # Render Sources Expander
                sources = retrieval_result.source_documents
                scores = retrieval_result.scores
                
                if sources:
                    with st.expander(f"📚 Retrieved Sources ({len(sources)})"):
                        for i, (doc, score) in enumerate(zip(sources, scores)):
                            source_name = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")
                            
                            st.markdown(f"**Source {i+1}**: `{source_name}` (Page {page}) — *Score: {score:.2f}*")
                            st.info(doc.page_content)
                else:
                    st.info("No matching sources found in the database.")
                    
            except Exception as e:
                logging.exception("Error during chat processing")
                st.error(f"An error occurred: {str(e)}")