'''
To be used in future if necessary.
'''

from app.llm.llm_handler import LLMHandler


class QueryRewriter:
    """
    Improves retrieval while preserving original intent.
    """

    def __init__(self, llm: LLMHandler):
        self.llm = llm

    def rewrite(self, query: str) -> str:

        prompt = f"""
You are a retrieval optimization assistant.

Your task:
- Improve the query for semantic retrieval
- Preserve the ORIGINAL meaning
- DO NOT answer the question
- DO NOT introduce unrelated domains or assumptions
- Keep the rewritten query concise

Original Query:
{query}

Rewritten Query:
"""

        rewritten_query = self.llm.llm.invoke(prompt).content.strip()

        # Fallback protection
        if len(rewritten_query.split()) > 25:
            return query

        return rewritten_query