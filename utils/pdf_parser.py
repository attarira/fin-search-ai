import pdfplumber
from typing import List, Dict
import os


def parse_pdf(file_path: str) -> List[Dict]:
    """
    Extracts text, page number, and paragraph from a PDF file.
    Returns a list of dicts: { 'doc_name', 'page', 'paragraph', 'text' }
    """
    doc_name = os.path.basename(file_path)
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                for para_num, para in enumerate(paragraphs, start=1):
                    chunks.append({
                        'doc_name': doc_name,
                        'page': page_num,
                        'paragraph': para_num,
                        'text': para
                    })
    return chunks
