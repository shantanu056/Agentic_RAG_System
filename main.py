from app.ingestion.loader import DocumentLoader

if __name__ == "__main__":
    loader = DocumentLoader("data/")
    docs = loader.load_documents()

    print(f"Loaded {len(docs)} documents\n")

    for i, doc in enumerate(docs[:2]):
        print(f"--- Document {i+1} ---")
        print(doc.page_content[:500])
        print("Metadata:", doc.metadata)
        print("\n")