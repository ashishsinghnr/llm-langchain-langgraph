"""
Sample 10: Agentic RAG — Agent-Driven Retrieval
=================================================
Combines Agents (04) + RAG (08) into the real-world retrieval pattern.
Instead of always retrieving, the agent *decides* when to search the
knowledge base vs answer from its own knowledge.

Key concepts:
  - Vector store as a tool the agent can call on demand
  - Agent reasoning: "Do I need to look this up, or do I already know?"
  - Multiple retrieval calls in a single question
  - Graceful fallback when the knowledge base doesn't have the answer

Pipeline:
  User Query → Agent decides → [search_docs tool] → Agent reads results → Answer
                    ↓ (or)
               Answer directly (no retrieval needed)

Why this matters:
  Basic RAG (08) always retrieves — even for "What is 2+2?". Agentic RAG
  is smarter: the agent only searches when the question needs it, saving
  latency and tokens. It can also search multiple times to gather more context.

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 10_agentic_rag.py
"""

import time
from config import get_openai_llm, get_embeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from openai import RateLimitError

import newrelic.agent

llm = get_openai_llm(temperature=0)
embeddings = get_embeddings()


# ===========================================================================
# Step 1: Build the knowledge base (same documents as 08_rag.py)
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

print("Building knowledge base...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)
chunks = splitter.create_documents(DOCUMENTS)
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print(f"  Ready — {len(chunks)} chunks indexed\n")


# ===========================================================================
# Step 2: Create a retrieval tool the agent can call
# ===========================================================================

@tool
def search_docs(query: str) -> str:
    """Search the internal knowledge base about LangChain, RAG, agents,
    embeddings, and LangGraph. Use this tool when the user asks about
    these topics. Do NOT use this for general knowledge questions."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(results)


# ===========================================================================
# Step 3: Create the agent with the retrieval tool
# ===========================================================================

agent = create_agent(
    llm,
    tools=[search_docs],
    system_prompt=(
        "You are a helpful assistant with access to a knowledge base about "
        "LangChain, RAG, embeddings, agents, and LangGraph. "
        "Use the search_docs tool when the user asks about these topics. "
        "For general knowledge questions (math, geography, etc.), answer "
        "directly without searching. Always cite which documents you used."
    ),
)


# ===========================================================================
# Step 4: Run examples
# ===========================================================================

def run_query(question: str):
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)
    result = agent.invoke({"messages": [("human", question)]})
    final = result["messages"][-1]
    print(f"\nAnswer: {final.content}")


def safe_run(question: str):
    for attempt in range(3):
        try:
            run_query(question)
            return
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n  [Rate limited — waiting {wait}s]")
            time.sleep(wait)
    print("  [Skipped — rate limit]")


nr_app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(nr_app, name="agentic-rag", group="LangChain"):

        # Agent SHOULD use the tool — topic is in the knowledge base
        safe_run("What is RAG and who introduced it?")

        # Agent SHOULD use the tool — topic is in the knowledge base
        safe_run("How does LangGraph differ from regular LangChain chains?")

        # Agent should answer DIRECTLY — general knowledge, not in docs
        safe_run("What is the capital of France?")

        # Agent SHOULD use the tool — needs to combine info from multiple chunks
        safe_run("What are the key components of LangChain and how do agents use tools?")

finally:
    newrelic.agent.shutdown_agent(timeout=10)
