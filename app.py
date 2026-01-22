import os
import streamlit as st
from ingestion.chunker import chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_index import FaissIndex


@st.cache_resource
def load_embedder():
    return Embedder()

embedder = load_embedder()

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


        chunks = chunk_documents(all_documents)

        st.write(f"Total chunks created: {len(chunks)}")

        st.write("Chunk preview:")
        for c in chunks[:3]:
            st.write(f"Chunk {c['chunk_id']} | Source: {c['source']} | Page: {c['page']}")
            st.write(c["text"][:300])
            st.markdown("---")


    embedder = Embedder()

    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    st.write(f"Generated embeddings for {len(embeddings)} chunks")
    st.write("Embedding vector length:", len(embeddings[0]))


    embedding_dim = len(embeddings[0])
    faiss_index = FaissIndex(embedding_dim)
    faiss_index.add_embeddings(embeddings)

    st.success("FAISS index built successfully")

query = st.text_input("Test semantic search (no LLM yet)")

if query:
    query_embedding = embedder.embed_texts([query])[0]
    distances, indices = faiss_index.search(query_embedding, top_k=5)

    st.write("Top matching chunks:")
    for idx, dist in zip(indices, distances):
        chunk = chunks[idx]
        st.write(f"Source: {chunk['source']} | Page: {chunk['page']} | Distance: {dist:.4f}")
        st.write(chunk["text"][:300])
        st.markdown("---")
