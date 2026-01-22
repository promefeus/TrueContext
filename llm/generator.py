def build_prompt(context_chunks, question):
    context_text = "\n\n".join(
        f"[Source: {c['source']} | Page: {c['page']}]\n{c['text']}"
        for c in context_chunks
    )

    prompt = f"""
You are an assistant answering questions strictly from the provided context.

Rules:
- Use ONLY the information in the context.
- You MAY summarize, list, or combine information if it is clearly present.
- Do NOT invent new terms, categories, or facts.
- If the answer cannot be reasonably derived from the context, say:
  "The answer is not available in the provided documents."

Context:
{context_text}

Question:
{question}

Answer (be concise and factual):
"""
    return prompt.strip()
