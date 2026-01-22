import os
import shutil
import streamlit as st

# Local Imports
from ingestion.loaders import load_document
from ingestion.chunker import chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_index import FaissIndex
# Added rewrite_query to imports
from llm.generator import build_prompt, rewrite_query 
from llm.groq_client import GroqLLM

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="TrueContext", 
    page_icon="🧠",
    layout="centered"
)

UPLOAD_DIR = "data/uploaded_docs"
TOP_K = 5
SIMILARITY_THRESHOLD = 1.6

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to answer questions based on your documents."}
    ]

# -------------------------------------------------
# MODEL LOADER
# -------------------------------------------------
@st.cache_resource
def load_resources():
    return Embedder(), GroqLLM()

try:
    embedder, llm = load_resources()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# -------------------------------------------------
# CORE LOGIC
# -------------------------------------------------
def cleanup_files():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

def process_documents(uploaded_files):
    saved_paths = []
    for file in uploaded_files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(path)

    all_docs = []
    for path in saved_paths:
        try:
            all_docs.extend(load_document(path))
        except Exception as e:
            st.error(f"Error reading {os.path.basename(path)}: {e}")
    
    cleanup_files()

    if not all_docs:
        return False, "No readable text found."

    chunks = chunk_documents(all_docs)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts)
    
    faiss_index = FaissIndex(len(embeddings[0]))
    faiss_index.add_embeddings(embeddings)

    return True, (chunks, faiss_index)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("TrueContext")
    st.markdown("#### 📂 Document Control")
    uploaded_files = st.file_uploader(
        "Upload Files", 
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Docs", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("No files selected.")
            else:
                with st.spinner("Analyzing content..."):
                    success, result = process_documents(uploaded_files)
                    if success:
                        chunks, index = result
                        st.session_state.chunks = chunks
                        st.session_state.faiss_index = index
                        st.session_state.indexed = True
                        st.session_state.messages = [{"role": "assistant", "content": "Documents processed! Ask me anything."}]
                        st.success(f"Indexed {len(chunks)} chunks.")
                    else:
                        st.error(result)

    with col2:
        if st.button("Clear All", type="secondary", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
    if st.session_state.indexed:
        st.info("✅ System Ready")

# -------------------------------------------------
# MAIN INTERFACE
# -------------------------------------------------
st.title("TrueContext")
st.markdown("#### Chat with your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.indexed:
            response = "Please upload documents first."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Thinking..."):
                # 1. Query Rewriting (Crucial Step)
                # We don't show this to the user, but we use it for search
                rewritten_query = rewrite_query(prompt)
                
                # 2. Embed the REWRITTEN query
                query_vec = embedder.embed_texts([rewritten_query])[0]
                dists, idxs = st.session_state.faiss_index.search(query_vec, top_k=TOP_K)
                
                # 3. Filtering Strategy
                relevant_chunks = []
                
                for dist, idx in zip(dists, idxs):
                    if dist < SIMILARITY_THRESHOLD:
                        if idx < len(st.session_state.chunks):
                            relevant_chunks.append(st.session_state.chunks[idx])

                # 4. Fallback Detection (Explicit Flag)
                is_fallback = False
                if len(relevant_chunks) == 0:
                    is_fallback = True
                    # Grab first 3 chunks (Introduction)
                    relevant_chunks = st.session_state.chunks[:3]

                # 5. Generate with Explicit Mode
                # We pass the ORIGINAL user prompt for the answer generation
                # but we use the rewrite for finding the chunks
                llm_prompt = build_prompt(relevant_chunks, prompt, is_fallback=is_fallback)
                
                answer = llm.generate(llm_prompt)
                st.markdown(answer)
                
                # Sources
                if relevant_chunks and not is_fallback:
                    with st.expander("View Reference Context"):
                        for c in relevant_chunks:
                            st.caption(f"Page {c['page']} | {os.path.basename(c['source'])}")
                            st.text(c['text'][:150] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})