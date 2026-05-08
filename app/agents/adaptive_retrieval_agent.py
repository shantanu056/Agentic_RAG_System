from app.llm.llm_handler import LLMHandler


class AdaptiveRetrievalAgent:
    """
    Dynamically selects retrieval strategy
    based on query complexity.
    """

    def __init__(self, llm: LLMHandler):
        self.llm = llm

    def determine_strategy(self, query: str) -> dict:

        prompt = f"""
You are a retrieval strategy agent.

Classify the query complexity:

1. simple
   - short factual query
   - direct answer expected

2. complex
   - comparison
   - explanation
   - multi-part reasoning
   - detailed analysis

ONLY return:
simple
OR
complex

Query:
{query}
"""

        response = self.llm.llm.invoke(prompt).content.strip().lower()

        if "complex" in response:
            return {
                "top_k": 12,
                "rerank_k": 6
            }

        return {
            "top_k": 6,
            "rerank_k": 3
        }