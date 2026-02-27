# LangChain Learning App

A hands-on introduction to LangChain — run each sample in order to learn the core concepts.

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Edit .env and paste your OpenAI API key
```

## Samples (run in order)

| # | File | What You Learn |
|---|------|----------------|
| 1 | `01_basic_llm.py` | Initialize a Chat Model, Prompt Templates, invoke the LLM |
| 2 | `02_chains_lcel.py` | LCEL pipe syntax, Output Parsers (string & JSON), sequential chains |
| 3 | `03_memory_chat.py` | Conversation history, session management, multi-turn chat |

Run any sample:
```bash
python 01_basic_llm.py
```

## Core Concepts Covered

```
Prompt Template  ──▶  LLM (ChatOpenAI)  ──▶  Output Parser
       │                                            │
       └── variables like {topic}           plain string or JSON
```

- **Chat Models** — wrappers around LLM APIs (OpenAI, Anthropic, etc.)
- **Prompt Templates** — reusable, parameterized prompts with system/human roles
- **LCEL (LangChain Expression Language)** — compose components with `|` (pipe)
- **Output Parsers** — convert raw AI responses into strings, JSON, or Pydantic models
- **Memory / Chat History** — maintain conversation context across multiple turns

## New Relic AI Monitoring

All samples are instrumented with **New Relic Python Agent v11.5.0** for AI observability.

### What's auto-captured
- LLM calls (model, tokens, duration, cost)
- Chain executions and agent/tool invocations
- Prompt and response content
- Distributed traces across sequential chains

### Setup
1. Generate a config: `newrelic-admin generate-config YOUR_LICENSE_KEY newrelic.ini`
2. Ensure these settings are in `newrelic.ini`:
   ```ini
   ai_monitoring.enabled = true
   ai_monitoring.streaming.enabled = true
   ai_monitoring.record_content.enabled = true
   ```
3. Run any sample — New Relic initializes automatically via code:
   ```python
   import newrelic.agent
   newrelic.agent.initialize("newrelic.ini")
   ```
4. View results in **New Relic UI → AI Monitoring**

## Next Steps

After completing these samples, explore:
- **RAG** (Retrieval-Augmented Generation) — answer questions from your own documents
- **Agents & Tools** — let the LLM call external tools (search, calculator, APIs)
- **Streaming** — stream tokens to the user in real time
# llm-langchain-langgraph
