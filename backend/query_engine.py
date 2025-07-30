
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple
import requests

INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/faiss_metadata.json"


def get_query_embedding(query: str) -> List[float]:
    # Use FinBERT for query embedding (same as ingest)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("yiyanghkust/finbert-embedding")
    return model.encode(query)


def search_index(query: str, top_k: int = 5) -> List[Dict]:
    # Load FAISS index and metadata
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    # Get query embedding
    query_emb = np.array([get_query_embedding(query)]).astype('float32')
    D, I = index.search(query_emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        meta = metadata[idx]
        meta['score'] = float(score)
        results.append(meta)
    return results


def generate_answer(query: str, contexts: List[Dict]) -> Dict:
    # Compose context for LLM prompt
    context_str = "\n\n".join([
        f"Source: {c['doc_name']} (page {c['page']}, para {c['paragraph']})\n{c['text']}" for c in contexts
    ])
    prompt = (
        f"You are a financial document assistant.\n"
        f"Answer the following question using ONLY the provided sources.\n"
        f"Show citations for each fact.\n"
        f"Highlight the most relevant text.\n"
        f"\nQuestion: {query}\n\nSources:\n{context_str}\n\nAnswer:"
    )
    # Use Ollama local Llama model for generation
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1",
        "prompt": prompt,
        "options": {"temperature": 0.2, "num_predict": 512}
    }
    response = requests.post(ollama_url, json=payload)
    if response.status_code == 200:
        answer = response.json().get("response", "")
    else:
        answer = f"Error from Ollama: {response.text}"
    return {
        "answer": answer,
        "sources": [
            {
                "doc_name": c['doc_name'],
                "page": c['page'],
                "paragraph": c['paragraph'],
                "score": c['score'],
                "text": c['text']
            } for c in contexts
        ]
    }


def query_engine(query: str, top_k: int = 5) -> Dict:
    results = search_index(query, top_k)
    answer = generate_answer(query, results)
    return answer

if __name__ == "__main__":
    # Example usage
    q = input("Enter your financial question: ")
    result = query_engine(q)
    print("\nAnswer:\n", result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"- {src['doc_name']} (page {src['page']}, para {src['paragraph']}), score: {src['score']:.2f}")
