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

    def generate_response(
    self,
    query: str,
    context: str = "",
    memory_context: str = "") -> str:
        """
        Generates grounded response using retrieved context.
        """

        prompt = f"""
            You are a helpful AI assistant.

            STRICT RULES:
            1. Answer ONLY from provided context and conversation history.
            2. Do NOT hallucinate.
            3. If answer is unavailable, say "I don't know."
            4. Maintain conversational continuity.

            Conversation History:
            {memory_context}

            Retrieved Context:
            {context}

            Current User Query:
            {query}

            Answer:
            """

        response = self.llm.invoke(prompt)

        return response.content