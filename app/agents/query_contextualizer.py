from app.llm.llm_handler import LLMHandler


class QueryContextualizer:
    """
    Resolves conversational references
    using memory context.
    """

    def __init__(self, llm: LLMHandler):
        self.llm = llm

    def contextualize(
        self,
        query: str,
        memory_context: str
    ) -> str:

        prompt = f"""
You are a conversational query resolver.

Rewrite the query ONLY if it depends
on previous conversation context.

Examples:

Conversation:
User: What is chunking?

Query:
Why is it important?

Output:
Why is chunking important?

Conversation:
{memory_context}

Query:
{query}

Output:
"""

        response = self.llm.llm.invoke(prompt)

        return response.content.strip()