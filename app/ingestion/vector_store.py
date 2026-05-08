from typing import List
from langchain_core.documents import Document

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """
    Handles embedding generation and FAISS vector storage.
    """

    def __init__(self):
        # Using lightweight embedding model (good balance of speed + quality)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_store = None

    def create_vector_store(self, documents: List[Document]):
        """
        Converts documents into embeddings and stores them in FAISS index.
        """

        self.vector_store = FAISS.from_documents(
            documents,
            self.embedding_model
        )

        return self.vector_store

    def save_vector_store(self, path: str = "vector_store"):
        """
        Saves FAISS index locally.
        """

        if self.vector_store:
            self.vector_store.save_local(path)

    def load_vector_store(self, path: str = "vector_store"):
        """
        Loads existing FAISS index.
        """

        self.vector_store = FAISS.load_local(
            path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        return self.vector_store