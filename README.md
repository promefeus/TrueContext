# TrueContext: Chat with your documents

**TrueContext** is a privacy-first, retrieval-augmented generation (RAG) system designed to facilitate accurate, hallucination-free interactions with user-uploaded documents. Unlike standard chatbots, TrueContext enforces strict grounding by prioritizing document content and employing advanced fallback mechanisms when specific context is missing.

This project demonstrates a serverless, ephemeral architecture suitable for academic research and study assistance, leveraging local vectorization and cloud-based Large Language Models (LLM) for inference.

---

## Key Technical Features

1. **Hybrid Retrieval Architecture**

  - ***Semantic Search***: Utilizes FAISS (CPU) with L2 normalization to perform high-speed similarity searches on dense vector embeddings.
  - ***Query Rewriting Pipeline***: Intercepts user queries and employs an intermediate LLM call to rewrite them into keyword-rich, search-optimized queries before vectorization. This significantly improves retrieval accuracy for conversational inputs.
  - ***Contextual Fallback Mechanism***: Features an explicit "Fallback Mode". If vector similarity scores drop below a strict threshold (1.6), the system automatically retrieves the document's introductory chunks and informs the LLM to switch from "Fact Retrieval" to "General Summary" mode.

2. **Privacy-First Design**
   
   - ***Ephemeral Storage***: The system follows a strict zero-retention policy. Uploaded files are processed into memory (RAM) and immediately deleted from the disk. No user data persists after the session ends.
   - ***Local Embedding***: Document vectorization happens locally using SentenceTransformers, ensuring that raw data is not sent to third-party embedding APIs.

3. Structured Generation
   
   - ***Streaming Responses***: Implements token-level streaming for a responsive, real-time user experience.
   - ***Enforced Output Structure***: The LLM prompt is engineered to strictly output answers in a specific format: Summary, Key Details, and Explanation, ensuring consistency across queries.

---


## Technology Stack

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Orchestration** | Streamlit | Frontend UI and session state management. |
| **LLM Inference** | Groq API (Llama 3) | High-speed inference for generation and query rewriting. |
| **Embeddings** | SentenceTransformers | all-MiniLM-L6-v2 (384-dimensional vectors). |
| **Vector Store** | FAISS (CPU) | Efficient similarity search and clustering. |
| **Ingestion** | PyMuPDF, Python-Docx | Robust parsing for PDF and DOCX formats. |

---

## Architecture Workflow

1. **Ingestion**: User uploads documents (PDF/DOCX/TXT). The system extracts text and splits it into 500-token chunks with overlap.

2. **Indexing**: Chunks are embedded locally into 384-d vectors and added to a FAISS FlatL2 index. Physical files are deleted.

3. **Query Processing**:
    - User submits a query.
    - ***Rewriter***: LLM refines the query (e.g., "What about the battery?" $\to$ "What are the battery specifications?").

4. **Retrieval**:
    - Rewritten query is embedded and matched against the index
    - Logic Gate: If top matches have high distance (low similarity), Fallback Mode is triggered.

5. **Generation**: The LLM generates a structured response using the retrieved context (or fallback introduction) and streams it to the UI.

---

## Installation & Setup

1. **Clone the Repository**
    ```bash
    git clone https://github.com/TensorNaut/TrueContext.git
    cd TrueContext


2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt


3. **Configure Environment**: Create a .env file in the root directory and add your Groq API key:
    ```bash
    GROQ_API_KEY=your_groq_api_key_here


4. **Run the Application**
    ```bash
    streamlit run app.py


## Project Structure
```bash

    TrueContext/
    ├── app.py                 # Main Streamlit application entry point
    ├── requirements.txt       # Project dependencies
    ├── ingestion/
    │   ├── loaders.py         # Logic for parsing PDF, DOCX, and TXT
    │   └── chunker.py         # Text splitting algorithms
    ├── embeddings/
    │   └── embedder.py        # SentenceTransformer wrapper
    ├── vectorstore/
    │   └── faiss_index.py     # FAISS index management
    └── llm/
        ├── groq_client.py     # Groq API client with streaming support
        └── generator.py       # Prompt engineering and query rewriting logic
