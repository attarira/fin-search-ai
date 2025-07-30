# FinSearchAI

FinSearchAI is a fully local, open-source semantic search and question-answering system for financial documents.

## Features

- Ingest and parse financial PDFs
- Generate domain-specific embeddings using FinBERT (HuggingFace)
- Store embeddings and metadata in FAISS
- Semantic search and retrieval
- RAG-based answer generation using Llama 3.1 via Ollama (local)
- Chat UI with citations, confidence scores, and highlights
- Document upload and automated ingestion from the Streamlit UI

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
4. Pull the Llama 2 model: `ollama pull llama3.1`
5. Start Ollama (it runs as a background service)
6. Run the Streamlit app: `streamlit run frontend/streamlit_app.py`

## Usage

- Upload financial documents (PDFs) via the UI
- Ask questions in natural language
- Get answers with citations and confidence scores

## License

MIT
