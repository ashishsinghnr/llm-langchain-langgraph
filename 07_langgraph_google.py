"""
Sample 7G: Explicit LangGraph — Custom Graph with Conditional Routing (Google Gemini)
======================================================================================
Same as 07_langgraph.py but uses Google's Gemini model.

Demonstrates:
  - StateGraph with typed state (TypedDict)
  - Nodes as functions that transform state
  - Conditional edges for routing based on LLM classification
  - Graph compilation and execution

Scenario: A support assistant that classifies a user query (technical,
billing, or general), routes to a specialist node, then produces a
final answer.

    ┌──────────┐     ┌───────────┐     ┌────────────┐     ┌───────┐
    │ classify │────▶│ technical │────▶│ respond    │────▶│  END  │
    │          │     └───────────┘     │            │     └───────┘
    │          │     ┌───────────┐     │            │
    │          │────▶│  billing  │────▶│            │
    │          │     └───────────┘     │            │
    │          │     ┌───────────┐     │            │
    │          │────▶│  general  │────▶│            │
    └──────────┘     └───────────┘     └────────────┘

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 07_langgraph_google.py
"""

import time
from typing import TypedDict, Literal
from config import get_google_llm
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START

import newrelic.agent

llm = get_google_llm(temperature=0)


# ===========================================================================
# Step 1: Define the graph state
# ===========================================================================

class SupportState(TypedDict):
    query: str           # User's original question
    category: str        # Classified category: technical, billing, general
    context: str         # Specialist knowledge added by the routing node
    response: str        # Final response to the user


# ===========================================================================
# Step 2: Define node functions (each takes state, returns partial update)
# ===========================================================================

def classify_node(state: SupportState) -> dict:
    """Use the LLM to classify the user query into a category."""
    prompt = ChatPromptTemplate.from_template(
        "Classify this customer query into exactly one category.\n"
        "Categories: technical, billing, general\n\n"
        "Query: {query}\n\n"
        "Respond with ONLY the category name, nothing else."
    )
    chain = prompt | llm | StrOutputParser()
    category = chain.invoke({"query": state["query"]}).strip().lower()

    # Normalize to one of the three valid categories
    if "technical" in category:
        category = "technical"
    elif "billing" in category:
        category = "billing"
    else:
        category = "general"

    print(f"  [Classifier] → {category}")
    return {"category": category}


def technical_node(state: SupportState) -> dict:
    """Handle technical queries with specialized context."""
    print("  [Router] → Technical specialist")
    return {
        "context": (
            "You are a technical support specialist. "
            "Focus on troubleshooting steps, system requirements, "
            "and technical solutions. Be precise and methodical."
        )
    }


def billing_node(state: SupportState) -> dict:
    """Handle billing queries with specialized context."""
    print("  [Router] → Billing specialist")
    return {
        "context": (
            "You are a billing support specialist. "
            "Focus on pricing, invoices, payment methods, and refunds. "
            "Be empathetic and clear about policies."
        )
    }


def general_node(state: SupportState) -> dict:
    """Handle general queries."""
    print("  [Router] → General support")
    return {
        "context": (
            "You are a friendly general support agent. "
            "Help with product info, features, and getting started. "
            "Be welcoming and helpful."
        )
    }


def respond_node(state: SupportState) -> dict:
    """Generate the final response using the specialist context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{context}"),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": state["context"],
        "query": state["query"],
    })
    return {"response": response}


# ===========================================================================
# Step 3: Define the routing function for conditional edges
# ===========================================================================

def route_by_category(state: SupportState) -> Literal["technical", "billing", "general"]:
    """Route to the appropriate specialist based on classification."""
    return state["category"]


# ===========================================================================
# Step 4: Build and compile the graph
# ===========================================================================

graph = StateGraph(SupportState)

# Add nodes
graph.add_node("classify", classify_node)
graph.add_node("technical", technical_node)
graph.add_node("billing", billing_node)
graph.add_node("general", general_node)
graph.add_node("respond", respond_node)

# Add edges
graph.add_edge(START, "classify")

# Conditional routing from classify → specialist
graph.add_conditional_edges(
    "classify",
    route_by_category,
    {"technical": "technical", "billing": "billing", "general": "general"},
)

# All specialists → respond → END
graph.add_edge("technical", "respond")
graph.add_edge("billing", "respond")
graph.add_edge("general", "respond")
graph.add_edge("respond", END)

# Compile into a runnable
app_graph = graph.compile()


# ===========================================================================
# Step 5: Run examples
# ===========================================================================

def run_query(query: str):
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)
    result = app_graph.invoke({"query": query})
    print(f"\nCategory: {result['category']}")
    print(f"Response: {result['response']}")


def safe_run(query: str):
    for attempt in range(3):
        try:
            run_query(query)
            return
        except ChatGoogleGenerativeAIError as e:
            print(f"\n  [Error: {e}]")
            wait = 30 * (attempt + 1)
            print(f"  [Retrying in {wait}s]")
            time.sleep(wait)
    print("  [Skipped — rate limit]")


nr_app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(nr_app, name="langgraph-google", group="LangChain"):

        safe_run("My app crashes when I try to upload files larger than 10MB")

        safe_run("I was charged twice for my subscription last month")

        safe_run("What features does your product offer?")

finally:
    newrelic.agent.shutdown_agent(timeout=10)
