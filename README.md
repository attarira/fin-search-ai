# FinSearchAI

FinSearchAI is an LLM-powered semantic search and question-answering system for financial documents.

## Features
- Ingest and parse financial PDFs
- Generate embeddings and store in FAISS
- Semantic search and retrieval
- RAG-based answer generation
- Chat UI with citations, confidence scores, and highlights

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r finsearchai/requirements.txt`
3. Add your API keys to `finsearchai/.env`
4. Run the Streamlit app: `streamlit run finsearchai/frontend/streamlit_app.py`

## Folder Structure
See `SPEC.md` for details.

## License
MIT
