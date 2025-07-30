# FinSearchAI Project Specification

## Overview

FinSearchAI is a fully local, open-source semantic search and QA system for financial documents using FinBERT embeddings and Llama 3.1 via Ollama for answer generation.

## Folder Structure

finsearchai/
├── backend/
│ ├── ingest.py # Process and embed docs (FinBERT)
│ ├── query_engine.py # Retrieval + RAG logic (Llama 3.1 via Ollama)
│ └── app.py # FastAPI endpoints (optional)
├── frontend/
│ └── streamlit_app.py # Streamlit UI (upload, search, ingest)
├── data/
│ └── sample_docs/ # Financial PDFs
├── models/
│ └── embedding_model/ # Optional local embedding model
├── utils/
│ ├── pdf_parser.py
│ └── text_cleaner.py
├── requirements.txt
├── SPEC.md
├── README.md
└── .env # (not required for local models)

## Features

- PDF ingestion and parsing
- Domain-specific embedding generation (FinBERT)
- FAISS-based vector storage
- Semantic query engine
- RAG-based answer generation (Llama 3.1 via Ollama)
- Chat UI with citations and highlights
- Document upload and automated ingestion

## Tech Stack

- Python, Streamlit
- FAISS, FinBERT (HuggingFace), Llama 3.1 (Ollama)
- PyMuPDF, pdfplumber

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
3. Pull the Llama 2 model: `ollama pull llama3.1`
4. Start Ollama (it runs as a background service)
5. Run Streamlit UI: `streamlit run frontend/streamlit_app.py`

## Usage

- Upload financial documents (PDFs) via the UI
- Ask questions in natural language
- Get answers with citations and confidence scores
