import os
import streamlit as st

UPLOAD_DIR = "data/uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="TrueContext", layout="wide")
st.title("TrueContext")

uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

saved_files = []

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_files.append(file_path)

    st.success("Files saved to disk:")
    for path in saved_files:
        st.write(path)

from ingestion.loaders import load_document

all_documents = []

if uploaded_files:
    for path in saved_files:
        docs = load_document(path)
        all_documents.extend(docs)

    st.write("Extracted content preview:")
    for d in all_documents[:5]:
        st.write(f"Source: {d['source']} | Page: {d['page']}")
        st.write(d["text"][:300])
        st.markdown("---")
