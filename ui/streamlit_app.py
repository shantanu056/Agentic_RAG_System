import sys
from pathlib import Path


# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import streamlit as st

from app.orchestration.rag_pipeline import RAGPipeline


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Session State Initialization
# -----------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []


pipeline = st.session_state.pipeline


# -----------------------------
# Title
# -----------------------------
st.title(" Agentic RAG System")

st.markdown(
    """
Enterprise-style Agentic RAG system with:

- Adaptive Retrieval
- Memory
- Validation Agent
- Conversational Context
"""
)


# -----------------------------
# Display Chat History
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User Input
# -----------------------------
query = st.chat_input(
    "Ask something..."
)


if query:

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.spinner("Thinking..."):

        response = pipeline.run(query)

    answer = response["answer"]

    # Show assistant response
    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander(" System Details"):

            st.write(
                f"**Query Type:** "
                f"{response['query_type']}"
            )

            if response["strategy"]:
                st.write(
                    f"**Retrieval Strategy:** "
                    f"{response['strategy']}"
                )

            if response["validation"]:
                st.write("**Validation Report:**")

                st.text(
                    response["validation"]
                )

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })