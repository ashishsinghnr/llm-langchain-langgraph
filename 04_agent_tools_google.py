"""
Sample 4G: Agent with Tools (Google Gemini)
=============================================
Same as 04_agent_tools.py but uses Google's Gemini model.

Requires:
  - GOOGLE_API_KEY in your .env file

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 04_agent_tools_google.py
"""

import ast
import math
from datetime import datetime
from config import get_google_llm, run_with_retry
from langchain_core.tools import tool
from langchain.agents import create_agent

import newrelic.agent

llm = get_google_llm(temperature=0)


# ---------------------------------------------------------------------------
# Step 1: Define tools using the @tool decorator
# ---------------------------------------------------------------------------

_SAFE_NAMES = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round}
_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call, ast.Name,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)


def _safe_eval(expression: str) -> str:
    """Evaluate a math expression using AST validation (no raw eval)."""
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            raise ValueError(f"Unknown function: {node.id}")
    code = compile(tree, "<calc>", "eval")
    return str(eval(code, {"__builtins__": {}}, _SAFE_NAMES))  # noqa: S307


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result.
    Use this for any arithmetic: addition, subtraction, multiplication,
    division, exponents, square roots, etc.
    Examples: "2 + 2", "sqrt(144)", "15 * 3.14", "2 ** 10"
    """
    try:
        return f"Status: OK\nResult: {_safe_eval(expression)}"
    except Exception as e:
        return (
            f"Status: ERROR\nError: {e}\n"
            "Next: Check syntax — use operators (+, -, *, /, **) "
            "and functions sqrt(), pow(), abs(), round()."
        )


@tool
def get_current_time() -> str:
    """Return the current date and time. Use this when the user asks
    what time or date it is."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def word_analyzer(text: str) -> str:
    """Analyze text and return word count, character count, and sentence count.
    Use this when the user wants statistics about a piece of text."""
    words = len(text.split())
    chars = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return (
        f"Status: OK\n"
        f"Words: {words}\nCharacters: {chars}\nSentences: {sentences}"
    )


# ---------------------------------------------------------------------------
# Step 2: Create the agent
# ---------------------------------------------------------------------------

agent = create_agent(
    llm,
    tools=[calculate, get_current_time, word_analyzer],
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use them when needed. Always show your reasoning."
    ),
)


# ---------------------------------------------------------------------------
# Step 3: Run the agent with different questions
# ---------------------------------------------------------------------------
def run_question(question: str):
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)
    def _run():
        result = agent.invoke({"messages": [("human", question)]})
        final = result["messages"][-1]
        print(f"\nFinal Answer: {final.content}")
    run_with_retry(_run)


app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(app, name="agent-tools-google", group="LangChain"):

        run_question("What is 1547 * 23 + sqrt(625)?")

        run_question("What is today's date and time?")

        run_question(
            "Analyze this text: 'LangChain agents are powerful. "
            "They can use tools dynamically!' "
            "Then calculate the word count times 100."
        )

finally:
    newrelic.agent.shutdown_agent(timeout=10)
