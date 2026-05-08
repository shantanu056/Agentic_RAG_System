from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks retrieved documents using a cross-encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        """

        # Prepare pairs (query, document)
        pairs = [(query, doc.page_content) for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to documents
        scored_docs = list(zip(documents, scores))

        # Sort by score (descending)
        ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        # Return top-k documents
        return [doc for doc, _ in ranked_docs[:top_k]]