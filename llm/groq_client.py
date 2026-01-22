import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GroqLLM:
    def __init__(self, model="llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env file")

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Generates a grounded response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are TrueContext, an intelligent assistant. You answer user questions clearly and accurately using the provided document context."
                },
                {"role": "user", "content": prompt}
            ],
            # 0.3 provides a balance: creative enough to write good sentences, 
            # strict enough to stick to facts.
            temperature=0.3, 
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()