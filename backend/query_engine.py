import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/faiss_metadata.json"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_query_embedding(query: str) -> List[float]:
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


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
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )
    answer = response.choices[0].message.content
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
