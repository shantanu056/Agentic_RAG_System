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

    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning to fix formatting issues.
        """

        if not text:
            return ""

        # Replace line breaks with space
        text = text.replace("\n", " ")

        # Remove extra spaces
        text = " ".join(text.split())

        return text


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
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Clean extracted text
            cleaned_docs = []
            for doc in docs:
                cleaned_docs.append(
                    Document(
                        page_content=self._clean_text(doc.page_content),
                        metadata=doc.metadata
                    )
                )

            # If meaningful text exists → return
            if any(doc.page_content.strip() for doc in cleaned_docs):
                return cleaned_docs

        except Exception:
            pass  # fallback to OCR

        # OCR fallback for scanned PDFs
        images = convert_from_path(file_path)

        for i, img in enumerate(images):
            text = self._clean_text(pytesseract.image_to_string(img))

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

        cleaned_docs = []
        for doc in docs:
            cleaned_docs.append(
                Document(
                    page_content=self._clean_text(doc.page_content),
                    metadata=doc.metadata
                )
            )

        return cleaned_docs


    def _load_csv(self, file_path: str) -> List[Document]:
        """
        Uses LangChain CSV loader.
        Each row becomes a Document.
        """

        loader = CSVLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.page_content = self._clean_text(doc.page_content)

        return docs


    def _load_image(self, file_path: str) -> List[Document]:
        """
        OCR for standalone image files.
        """

        image = Image.open(file_path)
        text = self._clean_text(pytesseract.image_to_string(image))

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "image"
                }
            )
        ]