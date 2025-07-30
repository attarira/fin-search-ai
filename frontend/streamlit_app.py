import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from backend.query_engine import query_engine

st.set_page_config(page_title="FinSearchAI", layout="wide")
st.title("FinSearchAI: Financial Document Semantic Search")

st.markdown("""
Ask questions about your financial documents. Answers are generated using LLMs and retrieved from your indexed sources.
""")

query = st.text_input("Enter your financial question:", "What is the net income for 2023?")

if st.button("Search") and query:
    with st.spinner("Searching and generating answer..."):
        result = query_engine(query)
        st.markdown(f"### Answer\n{result['answer']}")
        st.markdown("---")
        st.markdown("### Sources")
        for src in result["sources"]:
            st.markdown(f"**{src['doc_name']}** (page {src['page']}, para {src['paragraph']})  ")
            st.markdown(f"> {src['text']}")
            st.markdown(f"Score: {src['score']:.2f}")
            st.markdown("---")
