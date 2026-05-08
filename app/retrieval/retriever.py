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

    '''
    #Sample Fallack to basic retrieval if adaptive retrieval is not implemented yet.

    def retrieve(self, query: str) -> List[Document]:
        """
        Given a query, returns top-k relevant documents.
        """

        results = self.retriever.invoke(query)

        return results
    '''

    #Adaptive Retrieval:
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieves relevant documents dynamically.
        """

        k = top_k if top_k else self.top_k

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        results = retriever.invoke(query)

        return results