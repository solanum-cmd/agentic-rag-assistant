# 📚 DocMind AI — Agentic RAG Assistant

A fully functional, modern Retrieval-Augmented Generation (RAG) application built with Python, LangChain, ChromaDB, and Streamlit. This application allows users to upload PDF documents, and chat with an AI assistant that answers questions grounded *strictly* in the provided context, complete with source citations.

## ✨ Features

- **Multi-PDF Support:** Extract text from multiple PDFs at once using PyMuPDF.
- **Robust Vector Store:** Persistent local embeddings using ChromaDB.
- **Agentic Memory:** Conversational memory handles follow-up questions contextually.
- **Streaming Responses:** Real-time typewriter effect in the Streamlit UI.
- **Strict Grounding:** The assistant will refuse to answer if the information is not in the documents.
- **Source Citations:** Every response includes expanders showing the exact text chunks and relevance scores used to generate the answer.
- **Multi-Provider Support:** Easily switch between OpenAI and Google Gemini for LLMs, or HuggingFace and OpenAI for embeddings.

## 🏗️ Architecture

```text
User ──(Streamlit)──> AgentRAG Orchestrator
                         │
      ┌──────────────────┴──────────────────┐
      │                                     │
[PDF Upload]                          [Chat Query]
      │                                     │
      ▼                                     ▼
 pdf_reader.py (PyMuPDF)              chat.py (Memory mgmt)
      │                                     │
      ▼                                     ▼
 text_splitter.py (Chunks)            rag.py (RAG Pipeline)
      │                                     │
      ▼                                     ▼
 embeddings.py (Vectorise)        retriever.py (Similarity Search)
      │                                     │
      └──────► vector_store.py (ChromaDB) ◄─┘
```

## 🚀 Installation

1. **Clone the repository** (if not already done).

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Copy the template file to `.env`:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and configure your API keys and provider preferences.

## ⚙️ Configuration Reference (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `google` or `openai` | `google` |
| `EMBEDDING_PROVIDER` | `huggingface` (free) or `openai` | `huggingface` |
| `GOOGLE_API_KEY` | Your Gemini API Key | - |
| `OPENAI_API_KEY` | Your OpenAI API Key | - |
| `CHUNK_SIZE` | Max characters per document chunk | `800` |
| `RETRIEVER_TOP_K` | Number of chunks to retrieve per query | `4` |

*See `.env.example` for all configurable options.*

## 💻 Usage

### Starting the Web UI
Run the Streamlit application:
```bash
streamlit run app.py
```

### Using the CLI (Testing)
You can test the pipeline without the web interface:
```bash
# Ingest a PDF
python main.py ingest path/to/document.pdf

# Ask a question
python main.py ask "What is the main topic?"

# Clear the database
python main.py reset
```

## 🛠️ Troubleshooting

- **`ValueError: Unknown LLM_PROVIDER`**: Ensure your `.env` file contains either `google` or `openai`.
- **API Key Errors**: Ensure you have added your API key to `.env` without surrounding quotes.
- **Empty Responses**: If the assistant says it doesn't know, it means the retrieved chunks did not pass the relevance threshold or did not contain the answer. Try adjusting `CHUNK_SIZE` or `RELEVANCE_SCORE_THRESHOLD` in `.env`.
