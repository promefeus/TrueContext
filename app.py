import os
import streamlit as st

from ingestion.loaders import load_document
from ingestion.chunker import chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_index import FaissIndex
from llm.generator import build_prompt
from llm.groq_client import GroqLLM

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
UPLOAD_DIR = "data/uploaded_docs"
SIMILARITY_THRESHOLD = 1.0
TOP_K = 5

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="TrueContext", layout="wide")
st.title("TrueContext")
st.write("A grounded document question-answering system (RAG)")

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

# -------------------------------------------------
# LOAD MODELS (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_embedder():
    return Embedder()

@st.cache_resource
def load_llm():
    return GroqLLM()

embedder = load_embedder()
llm = load_llm()

# -------------------------------------------------
# SECTION 1: DOCUMENT UPLOAD
# -------------------------------------------------
st.header("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

saved_paths = []

if uploaded_files:
    for file in uploaded_files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(path)

    st.success(f"{len(saved_paths)} file(s) uploaded successfully.")

# -------------------------------------------------
# SECTION 2: INDEXING
# -------------------------------------------------
st.header("⚙️ Index Documents")

if uploaded_files and st.button("Index Documents"):
    all_documents = []

    for path in saved_paths:
        all_documents.extend(load_document(path))

    if len(all_documents) == 0:
        st.error("No readable content found in documents.")
        st.stop()

    chunks = chunk_documents(all_documents)

    if len(chunks) == 0:
        st.error("Chunking failed.")
        st.stop()

    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    faiss_index = FaissIndex(len(embeddings[0]))
    faiss_index.add_embeddings(embeddings)

    st.session_state.chunks = chunks
    st.session_state.faiss_index = faiss_index
    st.session_state.indexed = True

    st.success("Documents indexed successfully.")
    st.write(f"Chunks indexed: {len(chunks)}")

# -------------------------------------------------
# GATE: REQUIRE INDEXING
# -------------------------------------------------
if not st.session_state.indexed:
    st.info("Upload and index documents before asking questions.")
    st.stop()

# -------------------------------------------------
# SECTION 3: QUESTION ANSWERING
# -------------------------------------------------
st.header("❓ Ask a Question")

query = st.text_input("Ask a question based on uploaded documents")

if not query:
    st.stop()

# -------------------------------------------------
# RETRIEVAL
# -------------------------------------------------
query_embedding = embedder.embed_texts([query])[0]
distances, indices = st.session_state.faiss_index.search(
    query_embedding, top_k=TOP_K
)

best_distance = distances[0]

if best_distance > SIMILARITY_THRESHOLD:
    st.warning(
        "I couldn’t find enough information in the uploaded documents to answer this question."
    )
    st.stop()

trusted_chunks = []

for idx, dist in zip(indices, distances):
    if dist <= SIMILARITY_THRESHOLD:
        trusted_chunks.append(st.session_state.chunks[idx])

if len(trusted_chunks) == 0:
    st.warning(
        "Relevant context was found, but it wasn’t strong enough to answer reliably."
    )
    st.stop()

# -------------------------------------------------
# LLM GENERATION (GUARDED)
# -------------------------------------------------
prompt = build_prompt(trusted_chunks, query)
answer = llm.generate(prompt)

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
st.subheader("Answer")
st.write(answer)

st.subheader("Sources")
for c in trusted_chunks:
    st.write(f"{c['source']} | Page {c['page']}")

# -------------------------------------------------
# RESET
# -------------------------------------------------
st.divider()
if st.button("Reset App"):
    st.session_state.clear()
    st.experimental_rerun()
