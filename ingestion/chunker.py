def count_tokens(text: str) -> int:
    return len(text.split())

def chunk_documents(documents, chunk_size=400, overlap=60):
    chunks = []
    chunk_id = 0

    for doc in documents:
        words = doc["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"]
            })

            chunk_id += 1
            start = end - overlap

    return chunks
