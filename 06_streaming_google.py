"""
Sample 6G: Streaming Responses (Google Gemini)
================================================
Same as 06_streaming.py but uses Google's Gemini model.

Teaches:
  - llm.stream() for real-time token output
  - Streaming through LCEL chains (prompt | llm | parser)
  - Collecting streamed chunks into a final result

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 06_streaming_google.py
"""

import time
from config import get_google_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

import newrelic.agent

llm = get_google_llm(temperature=0.7)


# ===========================================================================
# Step 1: Basic streaming — stream directly from the LLM
# ===========================================================================

def example_basic_stream():
    """Stream tokens directly from the LLM."""
    print("=" * 60)
    print("Example A: Basic LLM streaming (Gemini)")
    print("=" * 60)

    print("Response: ", end="", flush=True)
    for chunk in llm.stream("Explain what streaming is in 2 sentences."):
        print(chunk.content, end="", flush=True)
    print("\n")


# ===========================================================================
# Step 2: Streaming through an LCEL chain
# ===========================================================================

def example_chain_stream():
    """Stream through a prompt | llm | parser chain."""
    print("=" * 60)
    print("Example B: Streaming through an LCEL chain")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise teacher. Explain topics in 3-4 sentences."),
        ("human", "Explain {topic} in the context of {domain}."),
    ])

    chain = prompt | llm | StrOutputParser()

    print("Response: ", end="", flush=True)
    for token in chain.stream({
        "topic": "backpropagation",
        "domain": "neural networks",
    }):
        print(token, end="", flush=True)
    print("\n")


# ===========================================================================
# Step 3: Collecting streamed output into a final string
# ===========================================================================

def example_collect_stream():
    """Stream tokens while also collecting the full response."""
    print("=" * 60)
    print("Example C: Stream + collect full response")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template(
        "Write a 4-line poem about {subject}."
    )
    chain = prompt | llm | StrOutputParser()

    collected = []
    print("Streaming: ", end="", flush=True)
    for token in chain.stream({"subject": "Python programming"}):
        print(token, end="", flush=True)
        collected.append(token)
    print()

    full_response = "".join(collected)
    print(f"\nCollected length: {len(full_response)} characters")
    print(f"Word count: {len(full_response.split())}")


# ===========================================================================
# Step 4: Run all examples
# ===========================================================================

def safe_run(fn):
    for attempt in range(3):
        try:
            fn()
            return
        except ChatGoogleGenerativeAIError as e:
            print(f"\n  [Error: {e}]")
            wait = 30 * (attempt + 1)
            print(f"  [Retrying in {wait}s]")
            time.sleep(wait)
    print("  [Skipped — rate limit]")


app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(app, name="streaming-google", group="LangChain"):

        safe_run(example_basic_stream)
        safe_run(example_chain_stream)
        safe_run(example_collect_stream)

finally:
    newrelic.agent.shutdown_agent(timeout=10)
