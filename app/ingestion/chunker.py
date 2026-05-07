from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Splits documents into smaller chunks for better embedding and retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Recursive splitter preserves structure (paragraph → sentence → words)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents while preserving metadata.
        """

        chunked_docs = self.splitter.split_documents(documents)

        return chunked_docs