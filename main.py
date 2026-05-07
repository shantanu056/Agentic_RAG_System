from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker


if __name__ == "__main__":
    loader = DocumentLoader("data/")
    docs = loader.load_documents()

    print(f"Loaded {len(docs)} documents")

    # Chunking step
    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50
    )

    chunked_docs = chunker.split_documents(docs)

    print(f"After chunking: {len(chunked_docs)} chunks\n")

    for i, doc in enumerate(chunked_docs[:2]):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:500])
        print("Metadata:", doc.metadata)
        print("\n")