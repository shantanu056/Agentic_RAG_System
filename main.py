from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.ingestion.vector_store import VectorStoreManager
from app.retrieval import reranker
from app.retrieval.retriever import Retriever
from app.llm.llm_handler import LLMHandler
from app.retrieval.reranker import Reranker
from app.agents.query_classifier import QueryClassifier
from app.agents.response_validator import ResponseValidator

import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Step 1: Load documents
    loader = DocumentLoader("data/")
    docs = loader.load_documents()
    print(f"Loaded {len(docs)} documents")

    # Step 2: Chunk documents
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunked_docs = chunker.split_documents(docs)
    print(f"After chunking: {len(chunked_docs)} chunks")

    # Step 3: Create vector store
    vector_manager = VectorStoreManager()

    if os.path.exists("vector_store"):
        print("📂 Loading existing vector store...")
        vector_store = vector_manager.load_vector_store()
    else:
        print("⚡ Creating new vector store...")
        vector_store = vector_manager.create_vector_store(chunked_docs)
        vector_manager.save_vector_store()

        print("✅ Vector store created")

    # Step 4: Initialize retriever
    retriever = Retriever(vector_store, top_k=5)

    # Step 5: Initialize LLM handler (for future use in answer generation)
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = LLMHandler(groq_api_key)

    #Query Classifier Agent:
    classifier = QueryClassifier(llm)

    #Response Validator Agent:
    validator = ResponseValidator(llm)

    # Step 7: Query loop
    while True:
        query = input("\n🔍 Enter your query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        #Reranker:
        reranker = Reranker()
        
        #Query Classification using Agent:
        query_type = classifier.classify(query)
        print(f"\n Query Type: {query_type}")

        if query_type == "retrieval":

            # Retrieval pipeline
            retrieved_docs = retriever.retrieve(query)

            reranked_docs = reranker.rerank(
                query,
                retrieved_docs,
                top_k=5
            )

            context = ""

            for i, doc in enumerate(reranked_docs):
                context += f"""
            [Source {i+1}]
            {doc.page_content} """

            answer = llm.generate_response(query, context)

        else:
            # Conversational response
            answer = llm.llm.invoke(query).content
        
        
        print("\n Answer:\n")
        print(answer)

        # Validate the response
        validation_result = validator.validate(query, context, answer)
        print("\n Validation Report:\n")
        print(validation_result["validation_result"])

