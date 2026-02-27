# LLM Apps — Architecture Overview

> 13 Progressive Samples × 2 Providers (OpenAI + Google Gemini)
> Each sample builds on previous concepts, from basic LLM calls to advanced orchestration patterns.

## Shared Infrastructure

```
┌──────────────┐    ┌────────────────────┐    ┌────────────────────────┐
│  config.py   │    │   newrelic.ini      │    │   .env                 │
│              │    │                     │    │                        │
│ get_openai() │    │  BackgroundTask()   │    │  NERD_COMPLETION_*     │
│ get_google() │    │  register_app()     │    │  API tokens & URLs     │
│ get_embed()  │    │  shutdown_agent()   │    │                        │
└──────┬───────┘    └─────────┬──────────┘    └────────────┬───────────┘
       │                      │                            │
       └──────────────────────┼────────────────────────────┘
                              │
            NERD_COMPLETION Proxy (staging-service.nr-ops.net)
            ┌─────────────────┴──────────────────┐
            │  OpenAI API    │   Google Gemini    │
            │  (GPT-5)       │  (gemini-2.5-flash)│
            └────────────────┴────────────────────┘
```

---

## Layer 1: Foundations

```
  01 Basic LLM          02 Chains/LCEL         03 Memory/Chat
  ┌──────────┐          ┌──────────┐           ┌──────────┐
  │ Prompt → │          │ Prompt → │           │ History →│
  │ LLM →    │          │ LLM →    │           │ Prompt → │
  │ Answer   │          │ Parser → │           │ LLM →    │
  └──────────┘          │ Answer   │           │ Answer   │
  Single call           └──────────┘           └──────────┘
                        LCEL pipes             Multi-turn
```

| Sample | What it teaches |
|--------|----------------|
| **01** | Direct LLM invocation, prompt templates, temperature |
| **02** | LCEL pipe syntax (`prompt \| llm \| parser`), chaining |
| **03** | Conversation history, `ChatMessageHistory`, multi-turn |

---

## Layer 2: Agents & Tools

```
  04 Agent + Tools       05 Multi-Agent         06 Streaming
  ┌──────────────┐      ┌──────────────┐       ┌──────────────┐
  │   ┌─────┐    │      │ Researcher   │       │ LLM ──stream─│─▶ token
  │   │ LLM │◀─┐ │      │   ↓          │       │              │─▶ token
  │   └──┬──┘  │ │      │ Writer       │       │ Chain ─stream│─▶ token
  │      ↓     │ │      │   ↓          │       │              │─▶ token
  │   @tools ──┘ │      │ Editor       │       └──────────────┘
  │  (ReAct loop)│      │   ↓ Answer   │       Token-by-token
  └──────────────┘      └──────────────┘
  Tool calling          Supervisor pattern
```

| Sample | What it teaches |
|--------|----------------|
| **04** | `@tool` decorator, `create_agent()`, ReAct loop |
| **05** | Multiple specialized agents, supervisor coordination |
| **06** | `llm.stream()`, chain streaming, collecting chunks |

---

## Layer 3: LangGraph & Retrieval

```
  07 StateGraph           08 RAG                  09 Structured Output
  ┌────────────────┐     ┌────────────────┐      ┌────────────────┐
  │ START          │     │ Docs → Split → │      │ LLM            │
  │  ↓             │     │ Embed → Store  │      │  ↓             │
  │ classify       │     │      ↓         │      │ Pydantic Model │
  │  ↓ (conditional)│    │ Query → Top-K →│      │  {title: str,  │
  │ tech/bill/gen  │     │ Context → LLM →│      │   rating: int, │
  │  ↓             │     │ Answer         │      │   ...}         │
  │ respond → END  │     └────────────────┘      └────────────────┘
  └────────────────┘     Retrieval chain         with_structured_output
  Conditional edges
```

| Sample | What it teaches |
|--------|----------------|
| **07** | `StateGraph`, `TypedDict` state, conditional edges, routing |
| **08** | Text splitting, embeddings, vector store, retrieval chain |
| **09** | Pydantic schemas, `with_structured_output()`, nested models |

---

## Layer 4: Advanced Patterns

```
  10 Agentic RAG          11 Human-in-the-Loop
  ┌─────────────────┐     ┌──────────────────────┐
  │ Agent            │     │ START → classify      │
  │  ↓               │     │          ↓            │
  │ "Do I need to    │     │   ┌─ sensitive? ─┐    │
  │  search?"        │     │   ↓              ↓    │
  │  YES → @tool ──┐ │     │ human_review  execute │
  │  NO  → answer  │ │     │ (interrupt)     ↓     │
  │  ↑             │ │     │   ↓          → END    │
  │  └─────────────┘ │     │ Command(resume)       │
  └─────────────────┘     └──────────────────────┘
  Agent decides when      interrupt() + resume
  to retrieve             InMemorySaver checkpoint
```

| Sample | What it teaches |
|--------|----------------|
| **10** | Agent + RAG combined, selective retrieval, `@tool` wrapping vector store |
| **11** | `interrupt()`, `Command(resume=...)`, `InMemorySaver`, `thread_id` |

---

## Layer 5: Orchestration

```
  12 Orchestrator-Worker              13 Evaluator-Optimizer
  ┌─────────────────────────┐        ┌─────────────────────────┐
  │                         │        │                         │
  │ orchestrator (plan)     │        │ ┌───────────┐           │
  │   ↓ Send() × N         │        │ │ generator │──────┐    │
  │ ┌────────┬────────┐    │        │ └───────────┘      ↓    │
  │ │worker 1│worker 2│... │        │      ▲        ┌─────────┐│
  │ └────┬───┴────┬───┘    │        │      │        │evaluator││
  │      ↓ reducer ↓       │        │      │        └────┬────┘│
  │ ┌──────────────────┐   │        │      │ feedback    │     │
  │ │   synthesizer    │   │        │      └─────────────┘     │
  │ └──────────────────┘   │        │  (loop until accepted    │
  │ Annotated[list, add]   │        │   or max iterations)     │
  └─────────────────────────┘        └─────────────────────────┘
  Dynamic parallelism                Self-correcting cycle
  LLM decides # of workers          Structured grading
```

| Sample | What it teaches |
|--------|----------------|
| **12** | `Send(node, arg)`, `Annotated[list, operator.add]` reducer, dynamic fan-out/fan-in |
| **13** | Graph cycles, generator-evaluator loop, iteration limits, structured feedback |

---

## Concept Progression

```
Simple ──────────────────────────────────────────────▶ Complex

LLM Call → Chains → Memory → Tools → Agents → Graph → RAG
                                                  ↓
                    Feedback Loops ← Workers ← Interrupts ← Agentic RAG
```

---

## File Structure

```
llmapps/
├── config.py                          ← Shared: LLM factories + embeddings
├── newrelic.ini                       ← New Relic AI Monitoring config
├── requirements.txt                   ← Dependencies
├── .env                               ← API tokens (not committed)
├── ARCHITECTURE.md                    ← This file
│
├── 01_basic_llm.py                   ┐
├── 01_basic_llm_google.py            ┘ Layer 1: Foundations
├── 02_chains_lcel.py                 ┐
├── 02_chains_lcel_google.py          ┘
├── 03_memory_chat.py                 ┐
├── 03_memory_chat_google.py          ┘
│
├── 04_agent_tools.py                 ┐
├── 04_agent_tools_google.py          ┘ Layer 2: Agents & Tools
├── 05_multi_agent.py                 ┐
├── 05_multi_agent_google.py          ┘
├── 06_streaming.py                   ┐
├── 06_streaming_google.py            ┘
│
├── 07_langgraph.py                   ┐
├── 07_langgraph_google.py            ┘ Layer 3: LangGraph & Retrieval
├── 08_rag.py                         ┐
├── 08_rag_google.py                  ┘
├── 09_structured_output.py           ┐
├── 09_structured_output_google.py    ┘
│
├── 10_agentic_rag.py                 ┐
├── 10_agentic_rag_google.py          ┘ Layer 4: Advanced Patterns
├── 11_human_in_the_loop.py           ┐
├── 11_human_in_the_loop_google.py    ┘
│
├── 12_orchestrator_worker.py         ┐
├── 12_orchestrator_worker_google.py  ┘ Layer 5: Orchestration
├── 13_evaluator_optimizer.py         ┐
└── 13_evaluator_optimizer_google.py  ┘
```

---

## Running Any Sample

```bash
# With New Relic monitoring
NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python <sample>.py

# Without monitoring
python <sample>.py
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain-openai` | ChatOpenAI, OpenAIEmbeddings |
| `langchain-google-genai` | ChatGoogleGenerativeAI |
| `langgraph` | StateGraph, Send, interrupt, Command |
| `langchain-core` | Prompts, parsers, tools, vector stores |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter |
| `pydantic` | Structured output schemas |
| `newrelic` | AI Monitoring instrumentation |
