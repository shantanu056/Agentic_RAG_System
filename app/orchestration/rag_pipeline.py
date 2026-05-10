import os
from dotenv import load_dotenv

from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.ingestion.vector_store import VectorStoreManager
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker

from app.llm.llm_handler import LLMHandler

from app.agents.query_classifier import QueryClassifier
from app.agents.response_validator import ResponseValidator
from app.agents.memory_agent import MemoryAgent
from app.agents.adaptive_retrieval_agent import AdaptiveRetrievalAgent
from app.agents.query_contextualizer import QueryContextualizer


class RAGPipeline:
    """
    Central orchestration layer for the Agentic RAG system.
    """

    def __init__(self):

        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY")

        # ---------- LLM ----------
        self.llm = LLMHandler(groq_api_key)

        # ---------- Agents ----------
        self.classifier = QueryClassifier(self.llm)
        self.validator = ResponseValidator(self.llm)
        self.memory_agent = MemoryAgent()
        self.adaptive_agent = AdaptiveRetrievalAgent(self.llm)
        self.contextualizer = QueryContextualizer(self.llm)

        # ---------- Retrieval ----------
        self.reranker = Reranker()

        # ---------- Load Vector Store ----------
        self.vector_manager = VectorStoreManager()

        self.vector_store = self.vector_manager.load_vector_store()

        self.retriever = Retriever(
            self.vector_store,
            top_k=8
        )

    def run(self, query: str) -> dict:
        """
        Executes full agentic RAG pipeline.
        """

        # ---------- Query Classification ----------
        query_type = self.classifier.classify(query)

        # ---------- Conversational Flow ----------
        if query_type == "conversational":

            answer = self.llm.llm.invoke(query).content

            return {
                "answer": answer,
                "query_type": query_type,
                "strategy": None,
                "validation": None
            }

        # ---------- Adaptive Retrieval ----------
        
        memory_context = self.memory_agent.get_context()

        resolved_query = self.contextualizer.contextualize(query=query,memory_context=memory_context)

        strategy = self.adaptive_agent.determine_strategy(resolved_query)


        # ---------- Retrieve ----------
        retrieved_docs = self.retriever.retrieve(resolved_query,top_k=strategy["top_k"])

        # ---------- Rerank ----------
        reranked_docs = self.reranker.rerank(
            query,
            retrieved_docs,
            top_k=strategy["rerank_k"]
        )

        # ---------- Build Context ----------
        context = ""

        for i, doc in enumerate(reranked_docs):
            context += f"""
[Source {i+1}]
{doc.page_content}

"""

        # ---------- Memory ----------
        memory_context = self.memory_agent.get_context()

        # ---------- Generate Response ----------
        answer = self.llm.generate_response(
            query=query,
            context=context,
            memory_context=memory_context
        )

        # ---------- Validation ----------
        validation = self.validator.validate(
            query=query,
            context=context,
            answer=answer
        )

        # ---------- Store Memory ----------
        self.memory_agent.add_interaction(
            user_query=query,
            assistant_response=answer
        )

        return {
            "answer": answer,
            "query_type": query_type,
            "strategy": strategy,
            "validation": validation["validation_result"]
        }