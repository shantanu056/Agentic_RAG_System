from app.llm.llm_handler import LLMHandler


class ResponseValidator:
    """
    Validates generated responses against retrieved context.
    """

    def __init__(self, llm: LLMHandler):
        self.llm = llm

    def validate(
        self,
        query: str,
        context: str,
        answer: str
    ) -> dict:

        prompt = f"""
You are a response validation agent.

Your task:
Evaluate whether the answer is fully supported by the provided context.

Check:
1. Is the answer grounded in the context?
2. Does it hallucinate information?
3. Does it fully answer the query?

Return STRICTLY in this format:

Grounded: yes/no
Complete: yes/no
Hallucination: yes/no
Reason: short explanation

Context:
{context}

Query:
{query}

Answer:
{answer}
"""

        response = self.llm.llm.invoke(prompt).content

        return {
            "validation_result": response
        }