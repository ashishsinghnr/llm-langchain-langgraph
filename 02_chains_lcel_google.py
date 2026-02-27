"""
Sample 2G: Chains with LCEL (Google Gemini)
=============================================
Same as 02_chains_lcel.py but uses Google's Gemini model instead of OpenAI.

Requires:
  - GOOGLE_API_KEY in your .env file  (get one at https://aistudio.google.com/apikey)

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 02_chains_lcel_google.py
"""

import time
from config import get_google_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

import newrelic.agent

llm = get_google_llm(temperature=0.7)


def invoke_with_retry(runnable, input_data, max_retries=3):
    """Invoke a LangChain runnable with retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data)
        except ChatGoogleGenerativeAIError as e:
            wait = 30 * (attempt + 1)
            print(f"  [Error: {e}]")
            print(f"  [Retrying in {wait}s — attempt {attempt + 1}/{max_retries}]")
            time.sleep(wait)
    print("  [Max retries reached, skipping this call]")
    return None


# Register with collector and create a transaction for this script
app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(app, name="chains-lcel-google", group="LangChain"):

        # -------------------------------------------------------------------
        # Example A: Simple Chain  (prompt | llm | parser)
        # -------------------------------------------------------------------
        print("=" * 60)
        print("Example A: Simple chain with pipe syntax (Gemini)")
        print("=" * 60)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a concise assistant. Answer in at most 2 sentences."),
            ("human", "Explain {concept} in the context of {domain}."),
        ])

        chain = prompt | llm | StrOutputParser()

        result = invoke_with_retry(chain, {
            "concept": "polymorphism",
            "domain": "object-oriented programming",
        })
        if result:
            print(result)
            print(f"\nType of result: {type(result)}")  # <class 'str'>

        # -------------------------------------------------------------------
        # Example B: Chain that returns structured JSON
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Example B: Chain with JSON output")
        print("=" * 60)

        json_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Always respond with valid JSON only, no markdown."),
            ("human",
             "Give me 3 {difficulty} quiz questions about {subject}. "
             "Return a JSON array of objects with keys: question, options (array of 4), answer."),
        ])

        json_chain = json_prompt | llm | JsonOutputParser()

        quiz = invoke_with_retry(json_chain, {
            "difficulty": "beginner",
            "subject": "Python",
        })

        if quiz:
            for i, q in enumerate(quiz, 1):
                print(f"\nQ{i}: {q['question']}")
                for opt in q["options"]:
                    print(f"    - {opt}")
                print(f"  Answer: {q['answer']}")

        # -------------------------------------------------------------------
        # Example C: Chaining two LLM calls (sequential chain)
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Example C: Sequential chain — one LLM feeds the next")
        print("=" * 60)

        outline_prompt = ChatPromptTemplate.from_template(
            "Write a one-paragraph story outline about: {topic}"
        )
        outline_chain = outline_prompt | llm | StrOutputParser()

        critique_prompt = ChatPromptTemplate.from_template(
            "Here is a story outline:\n\n{outline}\n\n"
            "Give 3 brief suggestions to improve it."
        )
        critique_chain = critique_prompt | llm | StrOutputParser()

        full_chain = (
            {"outline": outline_chain}
            | critique_chain
        )

        result = invoke_with_retry(full_chain, {"topic": "a robot learning to paint"})
        if result:
            print(result)

finally:
    newrelic.agent.shutdown_agent(timeout=10)
