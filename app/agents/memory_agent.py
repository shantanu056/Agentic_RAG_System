from typing import List


class MemoryAgent:
    """
    Stores and manages conversational memory.
    """

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history = []

    def add_interaction(self, user_query: str, assistant_response: str):
        """
        Stores user query and assistant response.
        """

        self.history.append({
            "user": user_query,
            "assistant": assistant_response
        })

        # Keep only recent interactions
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self) -> str:
        """
        Returns formatted conversation history.
        """

        conversation_context = ""

        for interaction in self.history:
            conversation_context += f"""
User: {interaction['user']}
Assistant: {interaction['assistant']}

"""

        return conversation_context