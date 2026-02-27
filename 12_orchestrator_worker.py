"""
Sample 12: Orchestrator-Worker — Dynamic Parallelism with Send API
===================================================================
The most powerful LangGraph pattern: an orchestrator LLM plans subtasks
at runtime, spawns parallel workers via Send(), and a synthesizer
combines the results.

Key concepts:
  - Send(node, arg) to dynamically create parallel worker tasks
  - Annotated[list, operator.add] reducer for accumulating worker outputs
  - Worker nodes receive isolated state (not the full graph state)
  - Number of workers is NOT predetermined — the LLM decides at runtime
  - Structured output for the orchestrator's planning step

Pipeline:
                                    ┌─────────────┐
                               ┌───▶│  worker (1)  │───┐
  ┌──────────────┐             │    └─────────────┘    │    ┌──────────────┐
  │ orchestrator │─── Send() ──┤    ┌─────────────┐    ├───▶│  synthesizer │──▶ END
  │ (plan)       │             ├───▶│  worker (2)  │───┤    │ (combine)    │
  └──────────────┘             │    └─────────────┘    │    └──────────────┘
                               │    ┌─────────────┐    │
                               └───▶│  worker (N)  │───┘
                                    └─────────────┘

Why this matters:
  Unlike parallelization with fixed nodes, the orchestrator-worker pattern
  lets the LLM decide HOW MANY subtasks to create and WHAT each should cover.
  This is how you build report generators, multi-file code editors, and
  research assistants that scale dynamically with the task.

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 12_orchestrator_worker.py
"""

import time
import operator
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from config import get_openai_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from openai import RateLimitError

import newrelic.agent

llm = get_openai_llm(temperature=0)


# ===========================================================================
# Step 1: Define schemas
# ===========================================================================

class Section(BaseModel):
    """A planned section of the report."""
    title: str = Field(description="Section title")
    description: str = Field(description="Brief description of what this section should cover")


class ReportPlan(BaseModel):
    """The orchestrator's plan: a list of sections to write."""
    sections: list[Section] = Field(description="List of report sections to write (3-4 sections)")


class ReportState(TypedDict):
    topic: str                                                  # User's topic
    sections: list[Section]                                     # Planned sections
    completed_sections: Annotated[list[str], operator.add]      # Reducer: workers append here
    final_report: str                                           # Synthesized output


# ===========================================================================
# Step 2: Define node functions
# ===========================================================================

def orchestrator_node(state: ReportState) -> dict:
    """Plan the report: decide what sections to write."""
    print(f"\n  [Orchestrator] Planning sections for: {state['topic']}")

    structured_llm = llm.with_structured_output(ReportPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research report planner. Given a topic, plan 3-4 sections "
         "for a concise report. Each section should have a clear title and a "
         "brief description of what it should cover. Keep it focused."),
        ("human", "Plan a report on: {topic}"),
    ])

    chain = prompt | structured_llm
    plan = chain.invoke({"topic": state["topic"]})

    for i, section in enumerate(plan.sections, 1):
        print(f"    Section {i}: {section.title}")

    return {"sections": plan.sections}


def worker_node(state: dict) -> dict:
    """Write a single section. Receives isolated state via Send()."""
    title = state["title"]
    description = state["description"]
    print(f"  [Worker] Writing: {title}")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research writer. Write a concise section for a report. "
         "Keep it to 3-4 sentences. Be informative and factual."),
        ("human",
         "Write a section titled '{title}'.\n"
         "It should cover: {description}"),
    ])

    chain = prompt | llm | StrOutputParser()
    content = chain.invoke({"title": title, "description": description})

    formatted = f"## {title}\n{content}"
    return {"completed_sections": [formatted]}


def synthesizer_node(state: ReportState) -> dict:
    """Combine all worker outputs into a final report."""
    print(f"\n  [Synthesizer] Combining {len(state['completed_sections'])} sections...")

    sections_text = "\n\n".join(state["completed_sections"])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a report editor. You are given individual sections of a report. "
         "Combine them into a cohesive final report. Add a brief introduction and "
         "conclusion. Keep the section content mostly intact but ensure smooth "
         "transitions. Output the final report in markdown."),
        ("human",
         "Topic: {topic}\n\n"
         "Sections:\n{sections}"),
    ])

    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({"topic": state["topic"], "sections": sections_text})

    return {"final_report": report}


# ===========================================================================
# Step 3: Dynamic dispatch with Send
# ===========================================================================

def assign_workers(state: ReportState) -> list[Send]:
    """Dispatch one worker per planned section."""
    return [
        Send("worker", {"title": s.title, "description": s.description})
        for s in state["sections"]
    ]


# ===========================================================================
# Step 4: Build and compile the graph
# ===========================================================================

graph = StateGraph(ReportState)

graph.add_node("orchestrator", orchestrator_node)
graph.add_node("worker", worker_node)
graph.add_node("synthesizer", synthesizer_node)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", assign_workers, ["worker"])
graph.add_edge("worker", "synthesizer")
graph.add_edge("synthesizer", END)

app_graph = graph.compile()


# ===========================================================================
# Step 5: Run examples
# ===========================================================================

def run_topic(topic: str):
    print(f"\n{'=' * 60}")
    print(f"Topic: {topic}")
    print("=" * 60)

    result = app_graph.invoke({"topic": topic})

    print(f"\n{'─' * 60}")
    print("FINAL REPORT:")
    print("─" * 60)
    print(result["final_report"])


def safe_run(topic: str):
    for attempt in range(3):
        try:
            run_topic(topic)
            return
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n  [Rate limited — waiting {wait}s]")
            time.sleep(wait)
    print("  [Skipped — rate limit]")


nr_app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(nr_app, name="orchestrator-worker", group="LangChain"):

        # Example A: AI topic — orchestrator decides sections dynamically
        safe_run("Artificial Intelligence in Healthcare")

        # Example B: Different topic — may produce different number of sections
        safe_run("History of Space Exploration")

finally:
    newrelic.agent.shutdown_agent(timeout=10)
