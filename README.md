# LangChain & LangGraph Learning App

A hands-on, progressive introduction to LangChain and LangGraph — 13 samples that build on each other, each available in both **OpenAI** and **Google Gemini** variants.

## Architecture Overview

```
                        ┌─────────────────────────────────────────────┐
                        │              config.py                      │
                        │  ┌─────────────┐  ┌──────────────────────┐  │
                        │  │ get_openai_  │  │ get_google_llm()     │  │
                        │  │ llm()        │  │ get_embeddings()     │  │
                        │  └─────────────┘  └──────────────────────┘  │
                        │  ┌─────────────────────────────────────────┐ │
                        │  │ invoke_with_retry() / run_with_retry() │ │
                        │  │ Centralized retry with lazy error       │ │
                        │  │ loading (OpenAI + Google errors)        │ │
                        │  └─────────────────────────────────────────┘ │
                        └──────────────────┬──────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
     ┌────────┴────────┐        ┌──────────┴──────────┐      ┌─────────┴─────────┐
     │  Basics (01-03) │        │  Agents (04-05)     │      │  LangGraph (07-13) │
     │                 │        │                     │      │                    │
     │  01 Basic LLM   │        │  04 Agent + Tools   │      │  07 StateGraph     │
     │  02 Chains/LCEL │        │  05 Multi-Agent     │      │  08 RAG            │
     │  03 Memory/Chat │        │                     │      │  09 Structured Out │
     └─────────────────┘        └─────────────────────┘      │  10 Agentic RAG   │
                                                             │  11 Human-in-Loop │
              ┌──────────────────────────────────────┐       │  12 Orch-Worker   │
              │  06 Streaming (standalone)           │       │  13 Eval-Optimizer │
              └──────────────────────────────────────┘       └───────────────────┘

     Each sample has two variants:
       xx_<name>.py         → OpenAI (GPT-5 via NERD proxy)
       xx_<name>_google.py  → Google Gemini (gemini-2.5-flash via NERD proxy)
```

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your NERD_COMPLETION_API_TOKEN and NERD_COMPLETION_BASE_URL
```

## Samples (run in order)

### Foundations

| # | File | What You Learn |
|---|------|----------------|
| 01 | `01_basic_llm.py` | Chat Models, Prompt Templates, basic LLM invocation |
| 02 | `02_chains_lcel.py` | LCEL pipe syntax, Output Parsers (string & JSON), sequential chains |
| 03 | `03_memory_chat.py` | Conversation history, session management, multi-turn chat |

### Agents & Tools

| # | File | What You Learn |
|---|------|----------------|
| 04 | `04_agent_tools.py` | `@tool` decorator, `create_agent()`, ReAct loop, AST-safe math evaluation |
| 05 | `05_multi_agent.py` | Specialized agents (research + writer), supervisor coordination pattern |

### Streaming

| # | File | What You Learn |
|---|------|----------------|
| 06 | `06_streaming.py` | Token-by-token streaming, `astream_events`, real-time output |

### LangGraph Patterns

| # | File | Pattern | What You Learn |
|---|------|---------|----------------|
| 07 | `07_langgraph.py` | Routing | `StateGraph`, conditional edges, LLM-driven routing |
| 08 | `08_rag.py` | RAG | Vector embeddings, retrieval, context-grounded generation |
| 09 | `09_structured_output.py` | Structured Output | Pydantic models, `with_structured_output()`, validation |
| 10 | `10_agentic_rag.py` | Agentic RAG | Agent-driven retrieval — search only when needed |
| 11 | `11_human_in_the_loop.py` | Human-in-the-Loop | `interrupt()` / `Command(resume=...)`, checkpointer |
| 12 | `12_orchestrator_worker.py` | Orchestrator-Worker | `Send()` API, dynamic parallelism, reducer pattern |
| 13 | `13_evaluator_optimizer.py` | Evaluator-Optimizer | Graph cycles, self-correcting feedback loop |

> Every sample has a `_google.py` variant (e.g., `04_agent_tools_google.py`) that runs the same pattern with Google Gemini.

### LangGraph Pattern Diagrams

**07 — Routing**
```
START ──▶ classify ──┬──▶ handle_technical ──▶ END
                     ├──▶ handle_billing   ──▶ END
                     └──▶ handle_general   ──▶ END
```

**10 — Agentic RAG**
```
User Query ──▶ Agent decides ──┬──▶ [search_docs tool] ──▶ Agent reads ──▶ Answer
                               └──▶ Answer directly (no retrieval)
```

**11 — Human-in-the-Loop**
```
START ──▶ classify ──▶ check_action ──┬──▶ execute_action ──▶ END
                                      └──▶ human_review (interrupt) ──▶ execute_action ──▶ END
```

**12 — Orchestrator-Worker**
```
                                  ┌─────────────┐
                             ┌───▶│  worker (1)  │───┐
┌──────────────┐             │    └─────────────┘    │    ┌──────────────┐
│ orchestrator │── Send() ──┤    ┌─────────────┐    ├───▶│  synthesizer │──▶ END
│ (plan)       │             ├───▶│  worker (2)  │───┤    │ (combine)    │
└──────────────┘             │    └─────────────┘    │    └──────────────┘
                             └───▶│  worker (N)  │───┘
                                  └─────────────┘
```

**13 — Evaluator-Optimizer**
```
┌─────────────┐     ┌─────────────┐
│  generator   │────▶│  evaluator   │───▶ END  (if accepted)
│  (write)     │◀────│  (grade)     │
└─────────────┘     └─────────────┘
      ▲ feedback          │
      └───────────────────┘  (if rejected, loop back)
```

## Project Structure

```
├── config.py                  # Centralized LLM clients, embeddings, retry utilities
├── 01_basic_llm.py            # ... through 13_evaluator_optimizer.py
├── *_google.py                # Google Gemini variants of each sample
├── studio_graphs.py           # Clean graph exports for LangGraph Studio
├── langgraph.json             # LangGraph Studio configuration
├── requirements.txt           # Python dependencies
├── newrelic.ini               # New Relic agent config (gitignored)
├── .env                       # API tokens (gitignored)
└── .claude/skills/            # Claude Code skills for development
```

## Key Design Decisions

- **Centralized config** — All LLM setup and retry logic in `config.py`; sample files stay focused on teaching the pattern
- **Lazy error loading** — Retry utilities import OpenAI/Google error classes on demand, so you don't need both SDKs installed
- **AST-safe eval** — Calculator tools use `ast.parse` + whitelist validation instead of raw `eval()`
- **Structured tool observations** — Agent tools return `Status`/`Summary`/`Next` fields following the agent-harness-construction pattern
- **Dual-provider samples** — Every pattern works with both OpenAI and Google Gemini via the NERD_COMPLETION proxy

## New Relic AI Monitoring

All samples are instrumented with **New Relic Python Agent** for AI observability.

### What's auto-captured
- LLM calls (model, tokens, duration, cost)
- Chain executions and agent/tool invocations
- Prompt and response content
- Distributed traces across sequential chains

### Setup
1. Set `NEW_RELIC_LICENSE_KEY` in your `.env` file
2. Ensure these settings in `newrelic.ini`:
   ```ini
   ai_monitoring.enabled = true
   ai_monitoring.streaming.enabled = true
   ai_monitoring.record_content.enabled = true
   ```
3. Run with New Relic:
   ```bash
   NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 01_basic_llm.py
   ```
4. View results in **New Relic UI -> AI Monitoring**

## LangGraph Studio

Serve the interactive graphs locally:

```bash
langgraph dev
# Open: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

Available graphs: Agent Tools, LangGraph Router, Agentic RAG, Human-in-the-Loop, Orchestrator-Worker, Evaluator-Optimizer.
