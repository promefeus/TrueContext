from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load model
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        Embeds texts into vectors.
        CRITICAL: normalize_embeddings=True ensures vectors are unit length.
        This makes FAISS L2 distance behave like Cosine Similarity.
        """
        return self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)