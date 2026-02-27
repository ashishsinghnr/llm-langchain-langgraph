"""
Sample 11G: Human-in-the-Loop — LangGraph interrupt() (Google Gemini)
======================================================================
Same as 11_human_in_the_loop.py but uses Google's Gemini model.
The interrupt/resume logic is identical — it's a LangGraph feature,
not LLM-specific.

Key concepts:
  - interrupt() to pause graph execution and surface a value to the user
  - Command(resume=...) to resume execution with the human's decision
  - InMemorySaver checkpointer (required for interrupt to work)
  - thread_id config for stateful graph sessions

Pipeline:
         ┌──────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────┐
  START─▶│ classify  │────▶│ check_action │────▶│ execute_action │────▶│ END │
         └──────────┘     └──────────────┘     └────────────────┘     └─────┘
                               │                      ▲
                               │ (if sensitive)        │
                               ▼                       │
                          ┌──────────────┐             │
                          │ human_review │─────────────┘
                          │ (interrupt)  │  (approved → execute)
                          └──────────────┘  (denied → END with rejection)

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 11_human_in_the_loop_google.py
"""

import time
import uuid
from typing import TypedDict, Literal
from config import get_google_llm
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

import newrelic.agent

llm = get_google_llm(temperature=0)


# ===========================================================================
# Step 1: Define the graph state
# ===========================================================================

class OrderState(TypedDict):
    query: str           # User's original request
    action: str          # Classified action: status, refund, cancel
    is_sensitive: bool   # Whether action needs human approval
    approved: bool       # Human approval result
    response: str        # Final response to the user


# ===========================================================================
# Step 2: Define node functions
# ===========================================================================

def classify_node(state: OrderState) -> dict:
    """Use the LLM to classify the user's request into an action type."""
    prompt = ChatPromptTemplate.from_template(
        "Classify this customer request into exactly one action.\n"
        "Actions: status, refund, cancel\n\n"
        "- status: checking order status, tracking, delivery info\n"
        "- refund: requesting money back, refund, return\n"
        "- cancel: cancelling an order\n\n"
        "Request: {query}\n\n"
        "Respond with ONLY the action name, nothing else."
    )
    chain = prompt | llm | StrOutputParser()
    action = chain.invoke({"query": state["query"]}).strip().lower()

    # Normalize to valid action
    if "refund" in action:
        action = "refund"
    elif "cancel" in action:
        action = "cancel"
    else:
        action = "status"

    is_sensitive = action in ("refund", "cancel")
    print(f"  [Classifier] action={action}, sensitive={is_sensitive}")
    return {"action": action, "is_sensitive": is_sensitive}


def human_review_node(state: OrderState) -> dict:
    """Pause execution and ask a human to approve or deny the action."""
    print(f"  [Human Review] Pausing for approval...")

    # interrupt() pauses the graph and sends this value to the client
    decision = interrupt({
        "action": state["action"],
        "query": state["query"],
        "message": f"Agent wants to process a {state['action']} for: '{state['query']}'. Approve?",
    })

    # When resumed, 'decision' contains the human's response
    approved = decision.get("approved", False)
    print(f"  [Human Review] Decision received: {'APPROVED' if approved else 'DENIED'}")
    return {"approved": approved}


def execute_node(state: OrderState) -> dict:
    """Generate the final response based on action and approval status."""
    if state["is_sensitive"] and not state.get("approved", False):
        response = (
            f"Your {state['action']} request has been denied by a supervisor. "
            "Please contact support for further assistance."
        )
        print(f"  [Execute] Action denied")
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful customer support agent. "
             "Generate a brief, professional response confirming the action.\n"
             f"Action type: {state['action']}\n"
             f"Approved: {state.get('approved', 'N/A (non-sensitive)')}\n"
             "Keep it to 2-3 sentences."),
            ("human", "{query}"),
        ])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": state["query"]})
        print(f"  [Execute] Action processed")

    return {"response": response}


# ===========================================================================
# Step 3: Routing functions
# ===========================================================================

def route_by_sensitivity(state: OrderState) -> Literal["human_review", "execute"]:
    """Route sensitive actions to human review, others straight to execute."""
    return "human_review" if state["is_sensitive"] else "execute"


# ===========================================================================
# Step 4: Build and compile the graph
# ===========================================================================

graph = StateGraph(OrderState)

# Add nodes
graph.add_node("classify", classify_node)
graph.add_node("human_review", human_review_node)
graph.add_node("execute", execute_node)

# Add edges
graph.add_edge(START, "classify")

# Conditional routing: sensitive → human_review, non-sensitive → execute
graph.add_conditional_edges(
    "classify",
    route_by_sensitivity,
    {"human_review": "human_review", "execute": "execute"},
)

# After human review → always execute (execute_node handles denied case)
graph.add_edge("human_review", "execute")
graph.add_edge("execute", END)

# Compile with checkpointer (REQUIRED for interrupt to work)
checkpointer = InMemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)


# ===========================================================================
# Step 5: Run examples
# ===========================================================================

def run_scenario(label: str, query: str, human_decision: dict | None = None):
    """Run a scenario through the graph, handling interrupts."""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"Query: {query}")
    print("=" * 60)

    # Each scenario gets its own thread
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # First run — may hit an interrupt
    interrupted = False
    for chunk in app_graph.stream({"query": query}, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupted = True
            info = chunk["__interrupt__"][0]
            print(f"\n  ** INTERRUPTED **")
            print(f"  Interrupt value: {info.value}")
            print(f"  Waiting for human decision...")

    if interrupted and human_decision is not None:
        # Resume with the human's decision
        print(f"\n  >> Human responds: {human_decision}")
        for chunk in app_graph.stream(
            Command(resume=human_decision), config, stream_mode="updates"
        ):
            # Process remaining nodes
            if "execute" in chunk:
                pass  # Node ran, state updated

    # Get final state
    final_state = app_graph.get_state(config)
    response = final_state.values.get("response", "(no response)")
    print(f"\nFinal Response: {response}")


def safe_run(label: str, query: str, human_decision: dict | None = None):
    for attempt in range(3):
        try:
            run_scenario(label, query, human_decision)
            return
        except ChatGoogleGenerativeAIError as e:
            print(f"\n  [Error: {e}]")
            wait = 30 * (attempt + 1)
            print(f"  [Retrying in {wait}s]")
            time.sleep(wait)
    print("  [Skipped — rate limit]")


nr_app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(nr_app, name="human-in-the-loop-google", group="LangChain"):

        # Example A: Non-sensitive — flows straight through, no interrupt
        safe_run(
            "Example A: Non-sensitive action (no approval needed)",
            "What's the status of order #1234?",
        )

        # Example B: Sensitive + APPROVED — interrupt, then resume with approval
        safe_run(
            "Example B: Sensitive action — APPROVED by human",
            "I want a refund for order #5678, the item arrived damaged.",
            human_decision={"approved": True},
        )

        # Example C: Sensitive + DENIED — interrupt, then resume with denial
        safe_run(
            "Example C: Sensitive action — DENIED by human",
            "Cancel order #9012 immediately.",
            human_decision={"approved": False},
        )

finally:
    newrelic.agent.shutdown_agent(timeout=10)
