"""
Sample 3: Conversational Memory (Chat with History)
=====================================================
This teaches how to build a chatbot that remembers previous messages.

Key concepts:
  - ChatMessageHistory for storing conversation turns
  - MessagesPlaceholder for injecting history into prompts
  - RunnableWithMessageHistory to auto-manage memory in a chain

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 03_memory_chat.py
"""

import time
from config import get_openai_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from openai import RateLimitError

import newrelic.agent

llm = get_openai_llm(temperature=0.7)


def invoke_with_retry(runnable, input_data, max_retries=3, **kwargs):
    """Invoke a LangChain runnable with retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data, **kwargs)
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  [Rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries}]")
            time.sleep(wait)
    print("  [Max retries reached, skipping this call]")
    return None


# ---------------------------------------------------------------------------
# Step 1: Define a prompt with a placeholder for chat history
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly AI tutor. Keep answers concise (2-3 sentences). "
     "Reference earlier parts of the conversation when relevant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Step 2: Set up a session-based message store
# ---------------------------------------------------------------------------
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ---------------------------------------------------------------------------
# Step 3: Wrap the chain with message history management
# ---------------------------------------------------------------------------
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Register with collector and create a transaction for this script
app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(app, name="memory-chat-examples", group="LangChain"):

        # -------------------------------------------------------------------
        # Step 4: Simulate a multi-turn conversation
        # -------------------------------------------------------------------
        config = {"configurable": {"session_id": "user-123"}}

        conversation = [
            "Hi! I want to learn about Python lists.",
            "How do I add an item to one?",
            "What's the difference between append and extend?",
            "Can you summarize everything we discussed?",
        ]

        for user_msg in conversation:
            print(f"\nYou: {user_msg}")
            response = invoke_with_retry(
                chain_with_history, {"input": user_msg}, config=config,
            )
            if response:
                print(f"AI:  {response}")

        # -------------------------------------------------------------------
        # Step 5: Show that a different session starts fresh
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("New session (user-456) — no memory of previous chat")
        print("=" * 60)

        config2 = {"configurable": {"session_id": "user-456"}}
        response = invoke_with_retry(
            chain_with_history,
            {"input": "What have we talked about so far?"},
            config=config2,
        )
        if response:
            print("\nYou: What have we talked about so far?")
            print(f"AI:  {response}")

finally:
    newrelic.agent.shutdown_agent(timeout=10)
