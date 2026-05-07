import os
from typing import List

from langchain_core.documents import Document

# LangChain loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader
)

# OCR dependencies
import pytesseract
from PIL import Image
from pdf2image import convert_from_path


class DocumentLoader:
    """
    Production-grade document loader.
    Handles multiple file types and returns LangChain Document objects
    with proper metadata.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_documents(self) -> List[Document]:
        """
        Main orchestrator:
        - Iterates through files
        - Routes to specific loaders
        - Returns unified Document list
        """
        documents = []

        for file_name in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file_name)

            if file_name.endswith(".pdf"):
                documents.extend(self._load_pdf(file_path))

            elif file_name.endswith(".docx"):
                documents.extend(self._load_docx(file_path))

            elif file_name.endswith(".csv"):
                documents.extend(self._load_csv(file_path))

            elif file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                documents.extend(self._load_image(file_path))

        return documents


    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Handles PDF loading:
        - Uses LangChain PyPDFLoader for normal PDFs
        - Falls back to OCR for scanned PDFs
        """

        documents = []

        try:
            # Try extracting text using LangChain loader
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # If meaningful text exists → return
            if any(doc.page_content.strip() for doc in docs):
                return docs

        except Exception:
            pass  # fallback to OCR

        # OCR fallback for scanned PDFs
        images = convert_from_path(file_path)

        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": i,
                        "type": "scanned_pdf"
                    }
                )
            )

        return documents


    def _load_docx(self, file_path: str) -> List[Document]:
        """
        Uses LangChain Docx loader for Word documents.
        """

        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        return docs


    def _load_csv(self, file_path: str) -> List[Document]:
        """
        Uses LangChain CSV loader.
        Each row becomes a Document (better for retrieval granularity).
        """

        loader = CSVLoader(file_path)
        docs = loader.load()

        return docs


    def _load_image(self, file_path: str) -> List[Document]:
        """
        OCR for standalone image files.
        """

        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "image"
                }
            )
        ]