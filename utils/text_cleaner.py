import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Basic text cleaning: remove extra whitespace, normalize."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(chunks: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Splits paragraphs into chunks with overlap.
    Returns a list of dicts with metadata and chunked text.
    """
    chunked = []
    for chunk in chunks:
        words = chunk['text'].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            chunked.append({
                'doc_name': chunk['doc_name'],
                'page': chunk['page'],
                'paragraph': chunk['paragraph'],
                'text': clean_text(chunk_text)
            })
            if end == len(words):
                break
            start = end - overlap
    return chunked
