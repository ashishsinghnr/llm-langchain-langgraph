"""
Sample 8G: RAG — Retrieval Augmented Generation (Google Gemini)
================================================================
Same as 08_rag.py but uses Google's Gemini model for generation.
Embeddings use OpenAI via NERD_COMPLETION (the proxy supports the
OpenAI /embeddings endpoint).

Teaches:
  - Splitting documents into chunks
  - Generating embeddings (vector representations of text)
  - Storing and searching vectors in a vector store
  - Building a retrieval chain that finds relevant context and answers

Pipeline:
  Documents → Split → Embed → Vector Store
                                    ↓
  User Query → Retrieve top-K chunks → Prompt + Context → LLM → Answer

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 08_rag_google.py
"""

import time
from config import get_google_llm, get_embeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

import newrelic.agent

llm = get_google_llm(temperature=0)
embeddings = get_embeddings()


# ===========================================================================
# Step 1: Create sample documents (in production, load from files/DB/APIs)
# ===========================================================================

DOCUMENTS = [
    """LangChain is an open-source framework for building applications powered by
    large language models (LLMs). It was created by Harrison Chase in October 2022.
    LangChain provides modular abstractions for working with LLMs, including prompt
    templates, chains, agents, memory, and retrievers. It supports multiple LLM
    providers including OpenAI, Google, Anthropic, and open-source models.""",

    """Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses
    by retrieving relevant documents from a knowledge base before generating an answer.
    RAG was introduced by Facebook AI Research in 2020. The key advantage is that RAG
    allows models to access up-to-date or domain-specific information without retraining.
    RAG reduces hallucinations by grounding responses in actual source documents.""",

    """Vector embeddings are numerical representations of text that capture semantic
    meaning. Similar texts have embeddings that are close together in vector space.
    Common embedding models include OpenAI's text-embedding-3-small, Google's
    text-embedding-004, and open-source models like sentence-transformers. Embeddings
    are stored in vector databases like Pinecone, Chroma, FAISS, or Weaviate for
    efficient similarity search.""",

    """LangGraph is a library built on top of LangChain for creating stateful,
    multi-actor applications. It uses a graph-based approach where nodes represent
    computation steps and edges define the flow between them. LangGraph supports
    conditional edges, cycles, and human-in-the-loop patterns. It is ideal for
    building complex agent workflows that go beyond simple linear chains.""",

    """Agents in LangChain are systems that use LLMs to decide which tools to call
    and in what order. The most common pattern is ReAct (Reason + Act), where the
    LLM thinks step-by-step, calls a tool, observes the result, and repeats until
    it has an answer. Tools are Python functions decorated with @tool that the agent
    can invoke. Multi-agent systems coordinate multiple specialized agents.""",
]


# ===========================================================================
# Step 2: Split documents into smaller chunks
# ===========================================================================

print("Step 1: Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # Max characters per chunk
    chunk_overlap=50,     # Overlap between chunks for context continuity
    separators=["\n\n", "\n", ". ", " "],
)

chunks = splitter.create_documents(DOCUMENTS)
print(f"  Split {len(DOCUMENTS)} documents into {len(chunks)} chunks")


# ===========================================================================
# Step 3: Create embeddings and store in vector store
# ===========================================================================

print("\nStep 2: Creating embeddings and building vector store...")
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
print(f"  Vector store ready with {len(chunks)} embedded chunks")


# ===========================================================================
# Step 4: Build the RAG chain
# ===========================================================================

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question based ONLY on the "
     "provided context. If the context doesn't contain the answer, say "
     "\"I don't have enough information to answer that.\"\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


def rag_chain(question: str) -> str:
    """Run the full RAG pipeline: retrieve → format → generate."""
    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    # Format context from retrieved documents
    context = "\n\n".join(doc.page_content for doc in docs)

    # Show what was retrieved
    print(f"  Retrieved {len(docs)} chunks:")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"    [{i}] {preview}...")

    # Generate answer with Gemini
    chain = rag_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# ===========================================================================
# Step 5: Run example queries
# ===========================================================================

def run_query(question: str):
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)
    answer = rag_chain(question)
    print(f"\nAnswer: {answer}")


def safe_run(question: str):
    for attempt in range(3):
        try:
            run_query(question)
            return
        except ChatGoogleGenerativeAIError as e:
            print(f"\n  [Error: {e}]")
            wait = 30 * (attempt + 1)
            print(f"  [Retrying in {wait}s]")
            time.sleep(wait)
    print("  [Skipped — rate limit]")


nr_app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(nr_app, name="rag-google", group="LangChain"):

        safe_run("What is RAG and why does it reduce hallucinations?")

        safe_run("How do agents decide which tools to use?")

        safe_run("What is the difference between LangChain and LangGraph?")

        # This question is NOT in our documents — tests the guardrail
        safe_run("What is the capital of France?")

finally:
    newrelic.agent.shutdown_agent(timeout=10)
