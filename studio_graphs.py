"""
Studio Graphs — Clean exports for LangGraph Studio / Agent Chat UI
===================================================================
Exports compiled graphs from samples 04, 07, 10, 11, 12, 13 as
module-level variables that LangGraph Studio can serve.

Rules for Studio compatibility:
  - Export compiled graphs (no checkpointer — Studio injects one)
  - No side effects on import (no New Relic, no hardcoded runs)
  - Graphs using create_agent() support Chat Mode (MessagesState)
  - Graphs using StateGraph support Graph Mode (custom state)

Usage:
  langgraph dev       # starts server at http://127.0.0.1:2024
  Then open Studio:   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
"""

import math
import operator
from typing import TypedDict, Literal, Annotated
from datetime import datetime

from pydantic import BaseModel, Field
from config import get_openai_llm, get_embeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send, interrupt

llm = get_openai_llm(temperature=0)
llm_creative = get_openai_llm(temperature=0.7)


# ═══════════════════════════════════════════════════════════════════════
# Graph 1: Agent with Tools (from Sample 04)
# ═══════════════════════════════════════════════════════════════════════
# Chat Mode — ask it math questions, current time, text analysis

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result.
    Use this for any arithmetic: addition, subtraction, multiplication,
    division, exponents, square roots, etc.
    Examples: "2 + 2", "sqrt(144)", "15 * 3.14", "2 ** 10"
    """
    allowed = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


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
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"


agent_tools = create_agent(
    llm,
    tools=[calculate, get_current_time, word_analyzer],
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use them when needed. Always show your reasoning."
    ),
)


# ═══════════════════════════════════════════════════════════════════════
# Graph 2: Support Router (from Sample 07)
# ═══════════════════════════════════════════════════════════════════════
# Graph Mode — enter a support query, see classification + routing


class SupportState(TypedDict):
    query: str
    category: str
    context: str
    response: str


def _classify_node(state: SupportState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        "Classify this customer query into exactly one category.\n"
        "Categories: technical, billing, general\n\n"
        "Query: {query}\n\n"
        "Respond with ONLY the category name, nothing else."
    )
    chain = prompt | llm | StrOutputParser()
    category = chain.invoke({"query": state["query"]}).strip().lower()
    if "technical" in category:
        category = "technical"
    elif "billing" in category:
        category = "billing"
    else:
        category = "general"
    return {"category": category}


def _technical_node(state: SupportState) -> dict:
    return {"context": "You are a technical support specialist. Focus on troubleshooting steps."}


def _billing_node(state: SupportState) -> dict:
    return {"context": "You are a billing support specialist. Focus on pricing and refunds."}


def _general_node(state: SupportState) -> dict:
    return {"context": "You are a friendly general support agent. Help with product info."}


def _respond_node(state: SupportState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{context}"),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return {"response": chain.invoke({"context": state["context"], "query": state["query"]})}


def _route_by_category(state: SupportState) -> Literal["technical", "billing", "general"]:
    return state["category"]


_support_graph = StateGraph(SupportState)
_support_graph.add_node("classify", _classify_node)
_support_graph.add_node("technical", _technical_node)
_support_graph.add_node("billing", _billing_node)
_support_graph.add_node("general", _general_node)
_support_graph.add_node("respond", _respond_node)
_support_graph.add_edge(START, "classify")
_support_graph.add_conditional_edges(
    "classify", _route_by_category,
    {"technical": "technical", "billing": "billing", "general": "general"},
)
_support_graph.add_edge("technical", "respond")
_support_graph.add_edge("billing", "respond")
_support_graph.add_edge("general", "respond")
_support_graph.add_edge("respond", END)

support_router = _support_graph.compile()


# ═══════════════════════════════════════════════════════════════════════
# Graph 3: Agentic RAG (from Sample 10)
# ═══════════════════════════════════════════════════════════════════════
# Chat Mode — ask about LangChain, RAG, agents, or general knowledge

DOCUMENTS = [
    """LangChain is an open-source framework for building applications powered by
    large language models (LLMs). It was created by Harrison Chase in October 2022.
    LangChain provides modular abstractions for working with LLMs, including prompt
    templates, chains, agents, memory, and retrievers. It supports multiple LLM
    providers including OpenAI, Google, Anthropic, and open-source models.""",

    """Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses
    by retrieving relevant documents from a knowledge base before generating an answer.
    RAG was introduced by Facebook AI Research in 2020. The key advantage is that RAG
    allows models to access up-to-date or domain-specific information without retraining.
    RAG reduces hallucinations by grounding responses in actual source documents.""",

    """Vector embeddings are numerical representations of text that capture semantic
    meaning. Similar texts have embeddings that are close together in vector space.
    Common embedding models include OpenAI's text-embedding-3-small, Google's
    text-embedding-004, and open-source models like sentence-transformers. Embeddings
    are stored in vector databases like Pinecone, Chroma, FAISS, or Weaviate for
    efficient similarity search.""",

    """LangGraph is a library built on top of LangChain for creating stateful,
    multi-actor applications. It uses a graph-based approach where nodes represent
    computation steps and edges define the flow between them. LangGraph supports
    conditional edges, cycles, and human-in-the-loop patterns. It is ideal for
    building complex agent workflows that go beyond simple linear chains.""",

    """Agents in LangChain are systems that use LLMs to decide which tools to call
    and in what order. The most common pattern is ReAct (Reason + Act), where the
    LLM thinks step-by-step, calls a tool, observes the result, and repeats until
    it has an answer. Tools are Python functions decorated with @tool that the agent
    can invoke. Multi-agent systems coordinate multiple specialized agents.""",
]

_embeddings = get_embeddings()
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "],
)
_chunks = _splitter.create_documents(DOCUMENTS)
_vector_store = InMemoryVectorStore.from_documents(_chunks, _embeddings)
_retriever = _vector_store.as_retriever(search_kwargs={"k": 3})


@tool
def search_docs(query: str) -> str:
    """Search the internal knowledge base about LangChain, RAG, agents,
    embeddings, and LangGraph. Use this tool when the user asks about
    these topics. Do NOT use this for general knowledge questions."""
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(results)


agentic_rag = create_agent(
    llm,
    tools=[search_docs],
    system_prompt=(
        "You are a helpful assistant with access to a knowledge base about "
        "LangChain, RAG, embeddings, agents, and LangGraph. "
        "Use the search_docs tool when the user asks about these topics. "
        "For general knowledge questions (math, geography, etc.), answer "
        "directly without searching. Always cite which documents you used."
    ),
)


# ═══════════════════════════════════════════════════════════════════════
# Graph 4: Human-in-the-Loop (from Sample 11)
# ═══════════════════════════════════════════════════════════════════════
# Graph Mode — enter an order request, Studio shows interrupt for approval


class OrderState(TypedDict):
    query: str
    action: str
    is_sensitive: bool
    approved: bool
    response: str


def _hitl_classify(state: OrderState) -> dict:
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
    if "refund" in action:
        action = "refund"
    elif "cancel" in action:
        action = "cancel"
    else:
        action = "status"
    return {"action": action, "is_sensitive": action in ("refund", "cancel")}


def _hitl_review(state: OrderState) -> dict:
    decision = interrupt({
        "action": state["action"],
        "query": state["query"],
        "message": f"Agent wants to process a {state['action']} for: '{state['query']}'. Approve?",
    })
    approved = decision.get("approved", False) if isinstance(decision, dict) else bool(decision)
    return {"approved": approved}


def _hitl_execute(state: OrderState) -> dict:
    if state["is_sensitive"] and not state.get("approved", False):
        return {"response": f"Your {state['action']} request has been denied by a supervisor."}
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful customer support agent. Generate a brief, "
         "professional response confirming the action.\n"
         f"Action type: {state['action']}\n"
         f"Approved: {state.get('approved', 'N/A')}\n"
         "Keep it to 2-3 sentences."),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return {"response": chain.invoke({"query": state["query"]})}


def _hitl_route(state: OrderState) -> Literal["human_review", "execute"]:
    return "human_review" if state["is_sensitive"] else "execute"


_hitl_graph = StateGraph(OrderState)
_hitl_graph.add_node("classify", _hitl_classify)
_hitl_graph.add_node("human_review", _hitl_review)
_hitl_graph.add_node("execute", _hitl_execute)
_hitl_graph.add_edge(START, "classify")
_hitl_graph.add_conditional_edges(
    "classify", _hitl_route,
    {"human_review": "human_review", "execute": "execute"},
)
_hitl_graph.add_edge("human_review", "execute")
_hitl_graph.add_edge("execute", END)

# NO checkpointer — Studio injects one automatically
human_in_the_loop = _hitl_graph.compile()


# ═══════════════════════════════════════════════════════════════════════
# Graph 5: Orchestrator-Worker (from Sample 12)
# ═══════════════════════════════════════════════════════════════════════
# Graph Mode — enter a topic, see parallel report generation


class Section(BaseModel):
    title: str = Field(description="Section title")
    description: str = Field(description="Brief description of what this section should cover")


class ReportPlan(BaseModel):
    sections: list[Section] = Field(description="List of report sections to write (3-4 sections)")


class ReportState(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list[str], operator.add]
    final_report: str


def _orch_plan(state: ReportState) -> dict:
    structured_llm = llm.with_structured_output(ReportPlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research report planner. Given a topic, plan 3-4 sections "
         "for a concise report. Each section should have a clear title and description."),
        ("human", "Plan a report on: {topic}"),
    ])
    plan = (prompt | structured_llm).invoke({"topic": state["topic"]})
    return {"sections": plan.sections}


def _orch_worker(state: dict) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research writer. Write a concise section for a report. "
         "Keep it to 3-4 sentences. Be informative and factual."),
        ("human", "Write a section titled '{title}'.\nIt should cover: {description}"),
    ])
    content = (prompt | llm | StrOutputParser()).invoke({
        "title": state["title"], "description": state["description"],
    })
    return {"completed_sections": [f"## {state['title']}\n{content}"]}


def _orch_synthesize(state: ReportState) -> dict:
    sections_text = "\n\n".join(state["completed_sections"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a report editor. Combine the sections into a cohesive final report. "
         "Add a brief introduction and conclusion. Output in markdown."),
        ("human", "Topic: {topic}\n\nSections:\n{sections}"),
    ])
    report = (prompt | llm | StrOutputParser()).invoke({
        "topic": state["topic"], "sections": sections_text,
    })
    return {"final_report": report}


def _orch_assign(state: ReportState) -> list[Send]:
    return [Send("worker", {"title": s.title, "description": s.description})
            for s in state["sections"]]


_orch_graph = StateGraph(ReportState)
_orch_graph.add_node("orchestrator", _orch_plan)
_orch_graph.add_node("worker", _orch_worker)
_orch_graph.add_node("synthesizer", _orch_synthesize)
_orch_graph.add_edge(START, "orchestrator")
_orch_graph.add_conditional_edges("orchestrator", _orch_assign, ["worker"])
_orch_graph.add_edge("worker", "synthesizer")
_orch_graph.add_edge("synthesizer", END)

orchestrator_worker = _orch_graph.compile()


# ═══════════════════════════════════════════════════════════════════════
# Graph 6: Evaluator-Optimizer (from Sample 13)
# ═══════════════════════════════════════════════════════════════════════
# Graph Mode — enter email context, see feedback loop in real-time


class Evaluation(BaseModel):
    grade: Literal["acceptable", "needs_improvement"] = Field(
        description="Whether the email meets professional standards"
    )
    feedback: str = Field(
        description="Specific feedback on what to improve (if needs_improvement)"
    )


class EmailState(TypedDict):
    context: str
    draft: str
    feedback: str
    grade: str
    iteration: int


def _eval_generate(state: EmailState) -> dict:
    iteration = state.get("iteration", 0) + 1
    if state.get("feedback"):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional email writer. Revise the email below based "
             "on the feedback. Keep it concise (3-5 sentences), professional, clear.\n\n"
             "Previous draft:\n{draft}\n\nFeedback:\n{feedback}"),
            ("human", "Original request: {context}"),
        ])
        draft = (prompt | llm_creative | StrOutputParser()).invoke({
            "context": state["context"], "draft": state["draft"],
            "feedback": state["feedback"],
        })
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional email writer. Write a concise email "
             "(3-5 sentences). Be professional, clear, direct. Include a subject line."),
            ("human", "{context}"),
        ])
        draft = (prompt | llm_creative | StrOutputParser()).invoke({"context": state["context"]})
    return {"draft": draft, "iteration": iteration}


def _eval_evaluate(state: EmailState) -> dict:
    structured_eval = llm.with_structured_output(Evaluation)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict email quality evaluator. Grade on:\n"
         "1. Professional tone\n2. Clarity\n3. Conciseness (3-5 sentences)\n"
         "4. Completeness\n5. Has a subject line\n\n"
         "If ALL criteria met → 'acceptable'. If ANY fails → 'needs_improvement'.\n\n"
         "Original request: {context}"),
        ("human", "Email to evaluate:\n{draft}"),
    ])
    result = (prompt | structured_eval).invoke({
        "context": state["context"], "draft": state["draft"],
    })
    return {"grade": result.grade, "feedback": result.feedback}


MAX_ITERATIONS = 3


def _eval_route(state: EmailState) -> Literal["generator", "__end__"]:
    if state["grade"] == "acceptable":
        return END
    if state["iteration"] >= MAX_ITERATIONS:
        return END
    return "generator"


_eval_graph = StateGraph(EmailState)
_eval_graph.add_node("generator", _eval_generate)
_eval_graph.add_node("evaluator", _eval_evaluate)
_eval_graph.add_edge(START, "generator")
_eval_graph.add_edge("generator", "evaluator")
_eval_graph.add_conditional_edges(
    "evaluator", _eval_route,
    {"generator": "generator", END: END},
)

evaluator_optimizer = _eval_graph.compile()
