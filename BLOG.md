# Building LLM Applications with LangChain & LangGraph: What I Learned

**Author:** Ashish Singh

---

## Why I Explored This

LLM-powered applications are quickly becoming the next major category of software. Frameworks like **LangChain** and **LangGraph** are what developers reach for to build them — from simple chatbots to complex multi-agent orchestration systems. I wanted to understand these patterns deeply, hands-on, to see how modern AI applications actually work under the hood.

This blog captures the core concepts I learned, the architecture patterns that matter, and how observability transforms debugging LLM apps from guesswork into science.

---

## The Journey: From a Single LLM Call to Full Orchestration

Building LLM apps is a progressive skill. Each concept builds on the last:

```mermaid
flowchart TB
    L1["<b>1. Foundations</b>\nLLM calls, prompt templates,\nchains, conversational memory"]
    L2["<b>2. Agents & Tools</b>\nReAct reasoning loop, tool calling,\nmulti-agent coordination, streaming"]
    L3["<b>3. State Machines & Retrieval</b>\nExplicit graph routing, RAG pipeline,\nstructured output with Pydantic"]
    L4["<b>4. Production Patterns</b>\nHuman-in-the-loop approval gates,\nagent-driven retrieval"]
    L5["<b>5. Advanced Orchestration</b>\nDynamic parallelism (fan-out/fan-in),\nself-correcting feedback loops"]

    L1 --> L2 --> L3 --> L4 --> L5

    style L1 fill:#4A90D9,color:#fff
    style L2 fill:#3B7DD8,color:#fff
    style L3 fill:#9B59B6,color:#fff
    style L4 fill:#E8744F,color:#fff
    style L5 fill:#C0392B,color:#fff
```

Let me walk through the most important things I learned at each level.

---

## Core Concept 1: Everything is a Composable Pipe

The single most important idea in LangChain is **LCEL (LangChain Expression Language)**. Every component — prompts, LLMs, parsers — snaps together using the pipe (`|`) operator:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("human", "Explain {concept} in simple terms."),
])

# This is a "chain" — data flows left to right
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"concept": "vector embeddings"})
```

```mermaid
flowchart LR
    A["Prompt\nTemplate"] -->|"formats\nmessages"| B["LLM\n(GPT / Gemini)"]
    B -->|"AI\nmessage"| C["Output\nParser"]
    C --> D["Clean Result\n(string or JSON)"]

    style A fill:#4A90D9,color:#fff
    style B fill:#E8744F,color:#fff
    style C fill:#50B86C,color:#fff
    style D fill:#9B59B6,color:#fff
```

**What I learned:** This pipe syntax isn't just syntactic sugar — it creates a `RunnableSequence` that can be streamed, batched, retried, and traced as a single unit. You can even chain two LLM calls sequentially where the output of one feeds the input of another.

---

## Core Concept 2: Agents — When the LLM Controls the Flow

A chain is a **fixed pipeline** — you define the steps. An agent is **dynamic** — the LLM decides which tools to call and in what order using a reasoning loop:

```python
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Use for any arithmetic."""
    return str(eval(expression))

@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# The agent sees the tool docstrings and decides when to use them
agent = create_agent(llm, tools=[calculate, get_current_time])
```

```mermaid
flowchart TB
    Q["User: 'What is 1547 * 23?'"] --> LLM["LLM Reasons:\n'I need the calculate tool'"]
    LLM -->|"calls tool"| T["calculate('1547 * 23')\n→ 35581"]
    T -->|"result"| LLM2["LLM Reasons:\n'I have the answer'"]
    LLM2 --> A["Answer: 'The result is 35,581'"]

    style Q fill:#F5F5F5,color:#333
    style LLM fill:#E8744F,color:#fff
    style T fill:#4A90D9,color:#fff
    style LLM2 fill:#E8744F,color:#fff
    style A fill:#50B86C,color:#fff
```

**What I learned:** The `@tool` decorator's **docstring is critical** — the LLM reads it to decide when to use the tool. A vague docstring means the agent picks the wrong tool. Also, agents can chain multiple tools in sequence — asking "analyze this text then multiply the word count by 100" triggers two tool calls automatically.

---

## Core Concept 3: StateGraph — Explicit Control Over the Flow

LangGraph lets you build **state machines** where you define exactly how data flows. Each node is a function that reads and updates typed state, and edges define routing:

```python
class SupportState(TypedDict):
    query: str        # User's question
    category: str     # Classified: technical, billing, general
    context: str      # Specialist knowledge
    response: str     # Final answer

graph = StateGraph(SupportState)
graph.add_node("classify", classify_node)
graph.add_node("technical", technical_node)
graph.add_node("billing", billing_node)
graph.add_conditional_edges("classify", route_by_category)
```

```mermaid
flowchart LR
    S(("START")) --> C["Classify\nQuery"]
    C -->|"technical"| T["Technical\nSpecialist"]
    C -->|"billing"| B["Billing\nSpecialist"]
    C -->|"general"| G["General\nSupport"]
    T --> R["Generate\nResponse"]
    B --> R
    G --> R
    R --> E(("END"))

    style C fill:#E8744F,color:#fff
    style T fill:#4A90D9,color:#fff
    style B fill:#9B59B6,color:#fff
    style G fill:#50B86C,color:#fff
    style R fill:#F39C12,color:#fff
```

**What I learned:** The difference between `create_agent()` and `StateGraph` is like the difference between an ORM and raw SQL. `create_agent()` hides the graph behind a convenience wrapper. `StateGraph` gives you full control — and production apps need that control for conditional routing, error handling, and custom state management.

---

## Core Concept 4: RAG — Grounding LLMs in Real Data

**RAG (Retrieval Augmented Generation)** is the most common production LLM pattern. Instead of relying on the model's training data, you feed it relevant documents at query time:

```mermaid
flowchart TB
    subgraph Indexing ["Indexing (one-time)"]
        D["Documents"] --> SP["Split into\nChunks"]
        SP --> EM["Generate\nEmbeddings"]
        EM --> VS[("Vector\nStore")]
    end

    subgraph Query ["Query (per request)"]
        Q["User\nQuestion"] --> VS
        VS -->|"top-K similar\nchunks"| CTX["Build Context\nfrom chunks"]
        CTX --> P["Prompt:\nContext + Question"]
        P --> L["LLM"]
        L --> A["Grounded\nAnswer"]
    end

    style D fill:#4A90D9,color:#fff
    style VS fill:#9B59B6,color:#fff
    style L fill:#E8744F,color:#fff
    style A fill:#50B86C,color:#fff
    style Q fill:#F5F5F5,color:#333
```

```python
# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.create_documents(documents)

# Create vector store with embeddings
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Query: retrieve relevant chunks, then generate answer
docs = retriever.invoke("What is RAG?")
context = "\n".join(doc.page_content for doc in docs)
answer = (rag_prompt | llm | StrOutputParser()).invoke({
    "context": context, "question": "What is RAG?"
})
```

**What I learned:** RAG has three critical tuning knobs — **chunk size** (too large = noise, too small = lost context), **overlap** (maintains continuity between chunks), and **K** (number of retrieved chunks). Getting these wrong makes the LLM either hallucinate or miss relevant information.

---

## Core Concept 5: Production Patterns That Change Everything

### Human-in-the-Loop: Pause, Ask, Resume

Production agents must NOT take irreversible actions (refunds, deletions, sends) without a human check. LangGraph's `interrupt()` pauses the graph mid-execution and waits for human input:

```python
from langgraph.types import interrupt, Command

def human_review_node(state):
    # This PAUSES the entire graph and sends context to the user
    decision = interrupt({
        "message": f"Agent wants to process a {state['action']}. Approve?",
        "action": state["action"],
    })
    # Execution resumes here when human responds
    return {"approved": decision.get("approved", False)}

# Resume later with: Command(resume={"approved": True})
```

```mermaid
flowchart TB
    S(("START")) --> CL["Classify\nAction"]
    CL -->|"non-sensitive\n(status check)"| EX["Execute\nAction"]
    CL -->|"sensitive\n(refund / cancel)"| HR{"INTERRUPT\nHuman Review"}
    HR -->|"approved"| EX
    HR -->|"denied"| DN["Reject &\nNotify User"]
    EX --> EN(("END"))
    DN --> EN

    style CL fill:#E8744F,color:#fff
    style HR fill:#F39C12,color:#000
    style EX fill:#50B86C,color:#fff
    style DN fill:#C0392B,color:#fff
```

**What I learned:** `interrupt()` requires a **checkpointer** — the graph serializes its entire state to storage, pauses, and can resume hours later from that exact checkpoint. Each conversation needs a unique `thread_id`. This is how LangGraph achieves pause/resume without polling or external queues.

### Orchestrator-Worker: Dynamic Parallelism

The most powerful pattern I encountered. An LLM **plans** subtasks at runtime, spawns **parallel workers** via `Send()`, and a synthesizer combines results. The number of workers isn't hardcoded — the LLM decides:

```python
from langgraph.types import Send

def assign_workers(state):
    """LLM planned 3 sections? Spawn 3 workers. Planned 5? Spawn 5."""
    return [
        Send("worker", {"title": s.title, "description": s.description})
        for s in state["sections"]
    ]
```

```mermaid
flowchart TB
    S(("START")) --> O["Orchestrator\n(LLM plans subtasks)"]
    O -->|"Send()"| W1["Worker 1"]
    O -->|"Send()"| W2["Worker 2"]
    O -->|"Send()"| W3["Worker N"]
    W1 --> SY["Synthesizer\n(combine results)"]
    W2 --> SY
    W3 --> SY
    SY --> E(("END"))

    style O fill:#E8744F,color:#fff
    style W1 fill:#4A90D9,color:#fff
    style W2 fill:#4A90D9,color:#fff
    style W3 fill:#4A90D9,color:#fff
    style SY fill:#50B86C,color:#fff
```

**What I learned:** Each worker gets **isolated state** — only the data passed via `Send()`, not the full graph state. Results aggregate back using **reducers** (`Annotated[list, operator.add]`). This is how you build report generators, multi-file code editors, and research assistants that scale dynamically.

### Evaluator-Optimizer: Self-Correcting Loops

Single-pass LLM output is unreliable. The fix: one LLM generates, another evaluates. If rejected, feedback loops back for revision:

```python
class Evaluation(BaseModel):
    grade: Literal["acceptable", "needs_improvement"]
    feedback: str  # Specific, actionable feedback

MAX_ITERATIONS = 3  # Safety guard against infinite loops
```

```mermaid
flowchart LR
    S(("START")) --> G["Generator\n(creative, temp=0.7)"]
    G --> EV["Evaluator\n(strict, temp=0)"]
    EV -->|"needs_improvement\n+ specific feedback"| G
    EV -->|"acceptable\nOR max iterations"| EN(("END"))

    style G fill:#4A90D9,color:#fff
    style EV fill:#E8744F,color:#fff
```

**What I learned:** Two key design choices: (1) use **different temperatures** — creative (0.7) for the generator, deterministic (0) for the evaluator, and (2) always have a **MAX_ITERATIONS guard** — without it, a picky evaluator creates an infinite loop. The evaluator uses `with_structured_output()` to return a typed grade, not free-form text.

---

## Full Architecture: How It All Connects

```mermaid
flowchart TB
    subgraph Foundation ["Foundation Layer"]
        PT["Prompt Templates"] --> LCEL["LCEL Chains\n(pipe syntax)"]
        LCEL --> MEM["Conversational\nMemory"]
    end

    subgraph Agents ["Agent Layer"]
        TOOLS["Custom Tools\n(@tool decorator)"] --> REACT["ReAct Agent\n(reason + act loop)"]
        REACT --> MULTI["Multi-Agent\n(supervisor routing)"]
    end

    subgraph Graphs ["LangGraph Layer"]
        SG["StateGraph\n(typed state + edges)"] --> COND["Conditional\nRouting"]
        COND --> RAG["RAG Pipeline\n(retrieve + generate)"]
        RAG --> STRUCT["Structured Output\n(Pydantic schemas)"]
    end

    subgraph Production ["Production Patterns"]
        HITL["Human-in-the-Loop\n(interrupt / resume)"]
        OW["Orchestrator-Worker\n(dynamic Send)"]
        EO["Evaluator-Optimizer\n(feedback loop)"]
    end

    Foundation --> Agents
    Agents --> Graphs
    Graphs --> Production

    subgraph Observability ["Observability Layer"]
        NR["AI Monitoring\n(traces, tokens, latency)"]
    end

    Foundation -.->|"instrument"| NR
    Agents -.->|"instrument"| NR
    Graphs -.->|"instrument"| NR
    Production -.->|"instrument"| NR

    style Foundation fill:#EBF5FB,color:#333
    style Agents fill:#E8F8F5,color:#333
    style Graphs fill:#F5EEF8,color:#333
    style Production fill:#FDEDEC,color:#333
    style Observability fill:#FEF9E7,color:#333
```

---

## Why Observability Matters for LLM Applications

LLM applications are fundamentally different from traditional software. Here's what makes them hard to debug without observability:

| Challenge | Why It's Hard | What Observability Reveals |
|-----------|---------------|---------------------------|
| **Non-deterministic output** | Same input, different output every time | Track output variance across runs, identify temperature/prompt issues |
| **Hidden costs** | Token usage isn't visible in code | Per-request token counts, cost attribution, model comparison |
| **Unpredictable latency** | RAG retrieval + LLM generation = variable response times | Breakdown: embedding time vs. retrieval vs. generation |
| **Chain/agent complexity** | A single user request may trigger 2–10 LLM calls | Full distributed trace showing every LLM call, tool invocation, and routing decision |
| **Feedback loops** | Evaluator-optimizer may iterate 1–3 times | Iteration count per request, convergence tracking |
| **Dynamic parallelism** | Orchestrator spawns unknown number of workers | Fan-out visibility, per-worker latency, aggregation timing |

**The key learning:** Without observability, debugging an LLM app that "sometimes gives bad answers" is like debugging a distributed system with `print` statements. You need traces, metrics, and event correlation.

---

## Integrating with New Relic: Seeing Inside LLM Apps

New Relic's AI Monitoring automatically instruments LangChain and LangGraph applications. Here's what the integration looks like in practice:

### Setup

```python
# newrelic.ini
[newrelic]
ai_monitoring.enabled = true
ai_monitoring.streaming.enabled = true
ai_monitoring.record_content.enabled = true
```

```bash
# Run any LLM application with full instrumentation
NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python my_app.py
```

That's it. No code changes needed.

### What You See

```mermaid
flowchart LR
    subgraph App ["Your LLM Application"]
        A["Chain / Agent / Graph"]
    end

    subgraph NR ["New Relic AI Monitoring"]
        direction TB
        T["Distributed Traces\n(full call tree)"]
        E["LLM Events\n(per completion)"]
        M["Metrics\n(tokens, latency, errors)"]
        C["Content Capture\n(prompts & responses)"]
    end

    App -->|"auto-instrumented"| NR

    style App fill:#4A90D9,color:#fff
    style NR fill:#1CE783,color:#333
    style T fill:#fff,color:#333
    style E fill:#fff,color:#333
    style M fill:#fff,color:#333
    style C fill:#fff,color:#333
```

**For each LLM call, New Relic captures:**

- **LlmChatCompletionSummary** — model, temperature, token counts (prompt + completion), duration, finish reason
- **LlmChatCompletionMessage** — one event per message (system prompt, user input, assistant response)
- **Distributed trace spans** — showing the full call tree: chain → retrieval → embedding → LLM → parser

### What I Learned by Observing My Own Apps

1. **RAG is expensive** — a single RAG query triggers an embedding call + vector search + LLM generation. The embedding call is often 30-40% of total latency
2. **Agents are unpredictable** — a ReAct agent might call 1 tool or 5 tools for the same type of question. Traces reveal the reasoning path
3. **Feedback loops multiply costs** — an evaluator-optimizer that iterates 3 times uses 6x the tokens of a single-pass call. Observability shows which prompts consistently need multiple iterations
4. **Streaming changes the trace shape** — streaming responses appear as a single span that stays open until the last token. The agent wraps the iterator and records the aggregated result on stream completion

### Querying LLM Data in NRQL

```sql
-- Token usage by model
SELECT average(request.model), sum(token_count)
FROM LlmChatCompletionSummary
FACET request.model SINCE 1 hour ago

-- Slowest LLM calls
SELECT average(duration)
FROM LlmChatCompletionSummary
FACET name SINCE 1 hour ago

-- Error rate by chain type
SELECT percentage(count(*), WHERE error IS true)
FROM LlmChatCompletionSummary
FACET request.model SINCE 1 day ago
```

---

## Key Takeaways

1. **LCEL pipes are the atom of LangChain** — every application, no matter how complex, is built from `prompt | llm | parser` chains composed together

2. **StateGraph is what production uses** — explicit state machines with typed state, conditional edges, and clear routing beat hidden agent abstractions for anything beyond prototyping

3. **Three patterns define advanced LLM apps** — human-in-the-loop (`interrupt`/`resume`), dynamic parallelism (`Send`), and self-correcting loops (graph cycles). Master these and you can build anything

4. **Observability isn't optional** — LLM apps are non-deterministic, expensive, and opaque. Traces and token metrics are the only way to understand what's actually happening

5. **Provider-agnostic by design** — switching from OpenAI to Google Gemini changes the LLM initialization, not the application logic. The framework abstractions (chains, graphs, tools) work identically across providers

