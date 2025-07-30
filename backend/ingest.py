import os
import json
import faiss
import numpy as np
from typing import List, Dict
from utils.pdf_parser import parse_pdf
from utils.text_cleaner import chunk_text
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/sample_docs"
INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/faiss_metadata.json"

# Use FinBERT embedding model from HuggingFace
FINBERT_MODEL = "yiyanghkust/finbert-embedding"
model = SentenceTransformer(FINBERT_MODEL)

def get_embedding(text: str) -> np.ndarray:
    """Get embedding from FinBERT."""
    return model.encode(text)


def process_folder(folder_path: str = DATA_DIR):
    all_chunks = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(folder_path, fname)
            print(f"Parsing {fname}...")
            chunks = parse_pdf(fpath)
            chunked = chunk_text(chunks)
            all_chunks.extend(chunked)
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def build_faiss_index(chunks: List[Dict]):
    print("Generating embeddings with FinBERT...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")
    # Save metadata for retrieval
    with open(META_PATH, "w") as f:
        json.dump(chunks, f)
    print(f"Metadata saved to {META_PATH}")


def main():
    chunks = process_folder()
    build_faiss_index(chunks)

if __name__ == "__main__":
    main()
