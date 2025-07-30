# FinSearchAI Project Specification

## Overview
FinSearchAI is a semantic search and QA system for financial documents using LLMs and vector search.

## Folder Structure
finsearchai/
├── backend/
│   ├── ingest.py             # Process and embed docs
│   ├── query_engine.py       # Retrieval + RAG logic
│   └── app.py                # FastAPI endpoints (optional)
├── frontend/
│   └── streamlit_app.py      # Streamlit UI
├── data/
│   └── sample_docs/          # Financial PDFs
├── models/
│   └── embedding_model/      # Optional local embedding model
├── utils/
│   ├── pdf_parser.py
│   └── text_cleaner.py
├── requirements.txt
├── SPEC.md
├── README.md
└── .env                      # API keys

## Features
- PDF ingestion and parsing
- Embedding generation
- FAISS-based vector storage
- Semantic query engine
- RAG-based answer generation
- Chat UI with citations and highlights

## Tech Stack
- Python, FastAPI, LangChain, Streamlit
- FAISS, OpenAI, PyMuPDF, pdfplumber

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add API keys to `.env`
3. Run Streamlit UI: `streamlit run frontend/streamlit_app.py`

## Usage
- Upload financial documents
- Ask questions in natural language
- Get answers with citations and confidence scores
