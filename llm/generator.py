from llm.groq_client import GroqLLM

def rewrite_query(user_query: str) -> str:
    """
    Refines the user's query to be more search-friendly.
    """
    llm = GroqLLM() 
    prompt = f"""
    TASK: Rewrite the following user question to be a standalone, keyword-rich search query.
    
    RULES:
    1. Remove conversational filler (e.g., "tell me about", "can you explain").
    2. Focus on the core entities and concepts.
    3. Return ONLY the rewritten text. No quotes, no "Here is the rewritten query".
    
    ORIGINAL QUESTION: {user_query}
    
    REWRITTEN QUERY:
    """
    return llm.generate(prompt)

def build_prompt(context_chunks, question, is_fallback=False):
    """
    Constructs a structured RAG prompt with explicit fallback handling.
    """
    
    # 1. Context Formatting
    if context_chunks:
        context_text = "\n\n".join(
            f"[Page {c['page']}] {c['text']}" 
            for c in context_chunks
        )
    else:
        context_text = "No context available."

    # 2. Fallback Instructions
    fallback_instruction = ""
    if is_fallback:
        fallback_instruction = """
        CRITICAL NOTE: 
        The specific retrieval failed to find exact matches for this query. 
        The context provided below is the INTRODUCTION of the document.
        - If the user is asking for a summary/overview, use this context to provide a high-level summary.
        - If the user is asking for specific details NOT in this introduction, explicitly state: 
          "I couldn't find that specific detail in the document overview."
        """

    # 3. Final Prompt Construction
    prompt = f"""
    You are TrueContext, an intelligent document assistant.
    
    {fallback_instruction}
    
    INSTRUCTIONS:
    1. Answer the user's question using the Context provided below.
    2. Structure your answer as follows:
       - **Summary**: A 2-3 sentence direct answer.
       - **Key Details**: Bullet points extracting specific facts from the text.
       - **Explanation**: A simple explanation of the concept (if applicable).
    3. If the answer is not in the text, say so clearly. Do not hallucinate.
    4. If multiple interpretations exist, list them clearly.
    5. If examples are relevant, provide them.
    6. If a detailed answer is explicitly requested, provide it.

    ---------------------
    CONTEXT:
    {context_text}
    ---------------------
    
    USER QUESTION:
    {question}
    
    ANSWER:
    """
    return prompt.strip()