# Agentic RAG System

An enterprise-style Agentic Retrieval-Augmented Generation (RAG) system built using LangChain, FAISS, Groq LLMs, and modular AI agents.

---

# Features

## Core RAG Pipeline
- Multi-format document ingestion
- OCR support for scanned PDFs/images
- Recursive text chunking
- Embedding generation
- FAISS vector database
- Semantic retrieval
- Reranking pipeline

---

# Agentic Architecture

## Query Classifier Agent
Routes queries intelligently between:
- retrieval pipeline
- conversational pipeline

## Memory Agent
Maintains conversational continuity using session memory.

## Adaptive Retrieval Agent
Dynamically changes:
- retrieval depth
- reranking strategy
based on query complexity.

## Response Validator Agent
Checks:
- grounding
- hallucination
- completeness

---

# Tech Stack

## LLM
- Groq
- LLaMA 3.1 8B Instant

## Frameworks
- LangChain
- Sentence Transformers

## Vector Database
- FAISS

## OCR
- Tesseract OCR
- Poppler

---

# Project Structure

```text
app/
│
├── agents/
├── ingestion/
├── retrieval/
├── llm/
│
data/
main.py