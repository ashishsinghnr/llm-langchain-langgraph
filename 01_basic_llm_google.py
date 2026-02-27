"""
Sample 1G: Basic LLM Call with Prompt Templates (Google Gemini)
================================================================
Same as 01_basic_llm.py but uses Google's Gemini model instead of OpenAI.

Requires:
  - GOOGLE_API_KEY in your .env file  (get one at https://aistudio.google.com/apikey)

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 01_basic_llm_google.py
"""

import time
from config import get_google_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

import newrelic.agent

llm = get_google_llm(temperature=0.7)


def invoke_with_retry(runnable, input_data, max_retries=3):
    """Invoke a LangChain runnable with retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data)
        except ChatGoogleGenerativeAIError as e:
            print(f"  [Error: {e}]")
            wait = 30 * (attempt + 1)
            print(f"  [Retrying in {wait}s]")
            time.sleep(wait)
    print("  [Max retries reached, skipping this call]")
    return None


# Register with collector and create a transaction for this script
app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(app, name="basic-llm-google", group="LangChain"):

        # -------------------------------------------------------------------
        # Example A: Simple direct invocation
        # -------------------------------------------------------------------
        print("=" * 60)
        print("Example A: Direct LLM call (Gemini)")
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
    newrelic.agent.shutdown_agent(timeout=10)
