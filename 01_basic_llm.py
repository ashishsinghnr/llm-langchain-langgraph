"""
Sample 1: Basic LLM Call with Prompt Templates
================================================
This is the simplest LangChain application. It teaches:
  - How to initialize a Chat Model (ChatOpenAI)
  - How to create reusable Prompt Templates
  - How to invoke the model and read the response

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 01_basic_llm.py
"""

import time
from config import get_openai_llm
from langchain_core.prompts import ChatPromptTemplate
from openai import RateLimitError

import newrelic.agent

llm = get_openai_llm(temperature=0.7)


def invoke_with_retry(runnable, input_data, max_retries=3):
    """Invoke a LangChain runnable with retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data)
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  [Rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries}]")
            time.sleep(wait)
    print("  [Max retries reached, skipping this call]")
    return None


# Register with collector and create a transaction for this script
app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(app, name="basic-llm-examples", group="LangChain"):

        # -------------------------------------------------------------------
        # Example A: Simple direct invocation
        # -------------------------------------------------------------------
        print("=" * 60)
        print("Example A: Direct LLM call")
        print("=" * 60)

        response = invoke_with_retry(llm, "What are the 3 key benefits of using LangChain?")
        if response:
            print(response.content)

        # -------------------------------------------------------------------
        # Example B: Using a Prompt Template (reusable & parameterized)
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Example B: Prompt Template")
        print("=" * 60)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant who explains {topic} concepts in simple terms."),
            ("human", "{question}"),
        ])

        messages = prompt.invoke({
            "topic": "Python programming",
            "question": "What is a decorator and when should I use one?",
        })

        response = invoke_with_retry(llm, messages)
        if response:
            print(response.content)

        # -------------------------------------------------------------------
        # Example C: Different prompt, same template pattern
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Example C: Reusing the template with different inputs")
        print("=" * 60)

        messages = prompt.invoke({
            "topic": "Machine Learning",
            "question": "Explain overfitting in one paragraph.",
        })

        response = invoke_with_retry(llm, messages)
        if response:
            print(response.content)

finally:
    # Always flush data to New Relic, even if LLM calls failed
    newrelic.agent.shutdown_agent(timeout=10)
