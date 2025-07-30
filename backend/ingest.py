import os
import json
import faiss
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from utils.pdf_parser import parse_pdf
from utils.text_cleaner import chunk_text

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR = "data/sample_docs"
INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/faiss_metadata.json"

EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API."""
    response = client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


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
    print("Generating embeddings...")
    embeddings = [get_embedding(chunk['text']) for chunk in chunks]
    dim = len(embeddings[0])
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
