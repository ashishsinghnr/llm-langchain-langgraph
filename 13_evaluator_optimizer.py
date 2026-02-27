"""
Sample 13: Evaluator-Optimizer — Self-Correcting Feedback Loop
===============================================================
One LLM generates content, another evaluates it against criteria.
If the evaluator rejects the output, feedback loops back to the
generator for improvement. This repeats until the output passes
or a max iteration limit is hit.

Key concepts:
  - Graph cycle: generator → evaluator → (pass? END : generator)
  - Structured output for grading (Literal grade + feedback string)
  - Iteration tracking to prevent infinite loops
  - Separation of concerns: generator and evaluator have different prompts

Pipeline:
    ┌─────────────┐     ┌─────────────┐
    │  generator   │────▶│  evaluator   │───▶ END  (if accepted)
    │  (write)     │◀────│  (grade)     │
    └─────────────┘     └─────────────┘
          ▲ feedback          │
          └───────────────────┘  (if rejected)

Why this matters:
  Single-pass LLM output is often inconsistent. By adding an evaluator
  that checks quality, tone, or correctness, you create a self-healing
  system. This is how production AI achieves reliability — not by hoping
  the first attempt is good, but by iterating until it IS good.

Scenario: Professional Email Writer
  User provides context → generator writes email → evaluator checks
  tone, clarity, length, and professionalism → loops until acceptable.

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 13_evaluator_optimizer.py
"""

import time
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from config import get_openai_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from openai import RateLimitError

import newrelic.agent

llm = get_openai_llm(temperature=0.7)
eval_llm = get_openai_llm(temperature=0)  # Evaluator should be deterministic


# ===========================================================================
# Step 1: Define state and evaluation schema
# ===========================================================================

class EmailState(TypedDict):
    context: str         # User's request (who, what, why)
    draft: str           # Current email draft
    feedback: str        # Evaluator's feedback (empty on first pass)
    grade: str           # "acceptable" or "needs_improvement"
    iteration: int       # Loop counter


class Evaluation(BaseModel):
    """The evaluator's structured assessment."""
    grade: Literal["acceptable", "needs_improvement"] = Field(
        description="Whether the email meets professional standards"
    )
    feedback: str = Field(
        description="Specific feedback on what to improve (if needs_improvement)"
    )


# ===========================================================================
# Step 2: Define node functions
# ===========================================================================

def generator_node(state: EmailState) -> dict:
    """Write or revise the email based on context and optional feedback."""
    iteration = state.get("iteration", 0) + 1

    if state.get("feedback"):
        # Revision pass — incorporate evaluator feedback
        print(f"  [Generator] Revision #{iteration} (incorporating feedback)")
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional email writer. Revise the email below "
             "based on the feedback provided. Keep the email concise (3-5 sentences), "
             "professional, and clear.\n\n"
             "Previous draft:\n{draft}\n\n"
             "Feedback:\n{feedback}"),
            ("human", "Original request: {context}"),
        ])
        chain = prompt | llm | StrOutputParser()
        draft = chain.invoke({
            "context": state["context"],
            "draft": state["draft"],
            "feedback": state["feedback"],
        })
    else:
        # First pass — write from scratch
        print(f"  [Generator] Writing initial draft")
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional email writer. Write a concise email "
             "(3-5 sentences) based on the user's request. Be professional, "
             "clear, and direct. Include a subject line."),
            ("human", "{context}"),
        ])
        chain = prompt | llm | StrOutputParser()
        draft = chain.invoke({"context": state["context"]})

    return {"draft": draft, "iteration": iteration}


def evaluator_node(state: EmailState) -> dict:
    """Evaluate the email against professional standards."""
    print(f"  [Evaluator] Grading draft (iteration {state['iteration']})...")

    structured_eval = eval_llm.with_structured_output(Evaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict email quality evaluator. Grade the email below on:\n"
         "1. Professional tone (no slang, appropriate formality)\n"
         "2. Clarity (clear purpose, no ambiguity)\n"
         "3. Conciseness (3-5 sentences, not rambling)\n"
         "4. Completeness (addresses the original request)\n"
         "5. Has a subject line\n\n"
         "If ALL criteria are met, grade as 'acceptable'.\n"
         "If ANY criterion fails, grade as 'needs_improvement' and provide "
         "specific, actionable feedback.\n\n"
         "Original request: {context}"),
        ("human", "Email to evaluate:\n{draft}"),
    ])

    chain = prompt | structured_eval
    result = chain.invoke({"context": state["context"], "draft": state["draft"]})

    print(f"  [Evaluator] Grade: {result.grade}")
    if result.grade == "needs_improvement":
        print(f"  [Evaluator] Feedback: {result.feedback}")

    return {"grade": result.grade, "feedback": result.feedback}


# ===========================================================================
# Step 3: Routing — loop or finish
# ===========================================================================

MAX_ITERATIONS = 3


def route_after_evaluation(state: EmailState) -> Literal["generator", "__end__"]:
    """If acceptable or max iterations reached, end. Otherwise, loop back."""
    if state["grade"] == "acceptable":
        print(f"  [Router] Accepted!")
        return END
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"  [Router] Max iterations ({MAX_ITERATIONS}) reached, accepting as-is")
        return END
    print(f"  [Router] Rejected — sending back to generator")
    return "generator"


# ===========================================================================
# Step 4: Build and compile the graph
# ===========================================================================

graph = StateGraph(EmailState)

graph.add_node("generator", generator_node)
graph.add_node("evaluator", evaluator_node)

graph.add_edge(START, "generator")
graph.add_edge("generator", "evaluator")
graph.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {"generator": "generator", END: END},
)

app_graph = graph.compile()


# ===========================================================================
# Step 5: Run examples
# ===========================================================================

def run_example(label: str, context: str):
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"Context: {context}")
    print("=" * 60)

    result = app_graph.invoke({"context": context, "iteration": 0})

    print(f"\n{'─' * 60}")
    print(f"FINAL EMAIL (after {result['iteration']} iteration(s), grade: {result['grade']}):")
    print("─" * 60)
    print(result["draft"])


def safe_run(label: str, context: str):
    for attempt in range(3):
        try:
            run_example(label, context)
            return
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n  [Rate limited — waiting {wait}s]")
            time.sleep(wait)
    print("  [Skipped — rate limit]")


nr_app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(nr_app, name="evaluator-optimizer", group="LangChain"):

        # Example A: Straightforward request — likely passes on first try
        safe_run(
            "Example A: Simple meeting request",
            "Write an email to my manager Sarah asking to reschedule "
            "our 1-on-1 from Tuesday to Thursday due to a client conflict.",
        )

        # Example B: Trickier — casual tone that evaluator should push back on
        safe_run(
            "Example B: Complaint email (needs professional tone)",
            "Write an email to a vendor complaining that their latest "
            "software update broke our integration and we need an urgent fix. "
            "I'm really frustrated but need to keep it professional.",
        )

        # Example C: Complex request — multiple points to cover
        safe_run(
            "Example C: Project update with multiple stakeholders",
            "Write an email to the leadership team summarizing Q4 progress: "
            "launched the new API (2 weeks early), onboarded 3 enterprise clients, "
            "but the mobile app is delayed by 1 month due to a dependency on the "
            "design team. Need budget approval for a contractor to help.",
        )

finally:
    newrelic.agent.shutdown_agent(timeout=10)
