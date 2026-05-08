from langchain_groq import ChatGroq


class LLMHandler:
    """
    Handles LLM interaction using Groq.
    """

    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0  # deterministic answers
        )

    def generate_response(self, query: str, context: str="") -> str:
        """
        Generates grounded response using retrieved context.
        """

        prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.

Guidelines:
- Provide a COMPLETE explanation
- Combine information from multiple context chunks
- Be clear, structured and concise
- If not found, say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content