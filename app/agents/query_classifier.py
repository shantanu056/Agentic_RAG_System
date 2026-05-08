from app.llm.llm_handler import LLMHandler


class QueryClassifier:
    """
    Classifies user query type for intelligent routing.
    """

    def __init__(self, llm: LLMHandler):
        self.llm = llm

    def classify(self, query: str) -> str:

        prompt = f"""
Classify the user query into ONE of these categories:

1. retrieval
   - requires document lookup
   - factual/project/domain questions

2. conversational
   - greetings
   - casual chat
   - generic conversation

ONLY return the category name.

Query:
{query}
"""

        response = self.llm.llm.invoke(prompt)

        return response.content.strip().lower()