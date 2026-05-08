from typing import List
from langchain_core.documents import Document


class Retriever:
    """
    Handles query-based retrieval from vector store.
    """

    def __init__(self, vector_store, top_k: int = 8):
        self.vector_store = vector_store
        self.top_k = top_k

        # Convert vector store into retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Given a query, returns top-k relevant documents.
        """

        results = self.retriever.invoke(query)

        return results