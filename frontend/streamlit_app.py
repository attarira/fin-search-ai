import streamlit as st
import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from backend.query_engine import query_engine

st.set_page_config(page_title="FinSearchAI", layout="wide")
st.title("FinSearchAI: Financial Document Semantic Search")

st.markdown("""
Ask questions about your financial documents. Answers are generated using LLMs and retrieved from your indexed sources.
""")

# --- Document Upload Section ---
st.markdown("### Upload a Financial Document (PDF)")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
    save_path = os.path.join("../data/sample_docs", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}. Indexing now...")
    # Automate ingestion after upload
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
        from backend.ingest import process_folder, build_faiss_index
        chunks = process_folder("../data/sample_docs")
        build_faiss_index(chunks)
        st.success("Document indexed and ready for search!")
    except Exception as e:
        st.error(f"Error during ingestion: {e}")

st.markdown("---")

query = st.text_input("Enter your financial question:", "What is the net income for 2023?")

if st.button("Search") and query:
    # Check if index exists before searching
    if not os.path.exists("../data/faiss_index.bin") or not os.path.exists("../data/faiss_metadata.json"):
        st.error("No index found. Please upload and ingest documents first.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                result = query_engine(query)
                st.markdown(f"### Answer\n{result['answer']}")
                st.markdown("---")
                st.markdown("### Sources")
                for src in result["sources"]:
                    st.markdown(f"**{src['doc_name']}** (page {src['page']}, para {src['paragraph']})  ")
                    st.markdown(f"> {src['text']}")
                    st.markdown(f"Score: {src['score']:.2f}")
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error during search: {e}")
