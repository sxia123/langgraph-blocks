# LangGraph Blocks (Alpha)

> Structured, Scratch/Blockly-style visual agent orchestration that compiles cleanly to **LangGraph** (`StateGraph` and Functional API `@entrypoint`/`@task`).

[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Why Blocks for LangGraph?

Freeform node-and-wire canvases get tangled fast once workflows introduce **loops, conditional branching, and human-in-the-loop interrupts**—the exact things LangGraph is built for.

**LangGraph Blocks** replaces unconstrained arrows with nested, coarse-grained structured blocks:

- **Entry / Exit:** `On Message (Trigger)` (binds input key & checkpointer), `Final Reply`
- **Cognitive (LLM):** `Ask LLM`, `Agent Tool Loop` (enforces `max_steps >= 1`)
- **Control Flow Containers:** `If / Else Branch`, `Repeat Until (Bounded)` (enforces `max_iterations >= 1`), `Parallel Branches` (fan-out + barrier join)
- **Human-in-the-Loop:** `Human Approval` (compiles to `interrupt()` with `On Approved` / `On Rejected` slots)
- **Tools:** `Call Tool` (invokes registered tools with `{{variable}}` interpolation)

From one **Canonical JSON AST**, `langgraph-blocks` provides:
1. **Static Code Generation (One-Way "Eject"):** Lower blocks into clean, standalone Python using either **`StateGraph`** or **`@entrypoint` / `@task`**.
2. **In-Memory Runtime Builder:** Construct a live `CompiledStateGraph` directly from JSON via `build_state_graph(...)` without `exec()`.
3. **Zero-Dependency Local Studio UI:** Author nested workflows visually, inspect synthesized `TypedDict` state keys, watch live compiled Python update, and simulate HITL pause/resume locally.

---

## Quickstart

### Installation

Core compiler, validator, CLI, and Studio UI require **zero external dependencies** (Python 3.11+ stdlib):

```bash
pip install -e .
```

To execute compiled workflows against real LangGraph in memory:

```bash
pip install -e ".[runtime]"
```

### 1. Launch the Local Blocks Studio UI

```bash
langgraph-blocks studio
```

Opens `http://127.0.0.1:8842` (automatically falling back to another free port if `8842` is busy, avoiding `8000` and `8765`) with starter templates (`Support Refund + HITL`, `Research & Critique Loop`), live `StateGraph` / `Functional API` compilation, JSON import/export, and step-by-step HITL simulation.

### 2. CLI Workflow

```bash
# Scaffold a starter block workflow JSON
langgraph-blocks init flow.json --preset support_refund_hitl

# Validate block structure and loop/step safety bounds
langgraph-blocks validate flow.json

# Compile to standalone StateGraph Python code
langgraph-blocks compile flow.json --target stategraph -o flow_sg.py

# Or compile to LangGraph Functional API (@entrypoint / @task)
langgraph-blocks compile flow.json --target functional -o flow_func.py

# Simulate locally (auto-approving human_approval interrupts)
langgraph-blocks run flow.json --input "Please refund order #8492" --approve
```

### 3. Python API

```python
import json
from langgraph.types import Command
from langgraph_blocks import (
    build_state_graph,
    compile_to_functional_api,
    compile_to_state_graph,
)

with open("examples/support_refund_hitl.json", encoding="utf-8") as f:
    workflow = json.load(f)

# 1) Emit standalone Python code
sg_code = compile_to_state_graph(workflow)
func_code = compile_to_functional_api(workflow)

# 2) Or build a live LangGraph CompiledStateGraph directly in memory
graph = build_state_graph(
    workflow,
    llm=lambda prompt, state: "refund" if "refund" in state.get("user_query", "").lower() else "general",
    tools={
        "lookup_order": lambda arg, _s: {"order_id": "8492", "amount": "$49.00"},
        "issue_refund": lambda arg, _s: {"refund_id": "RF-991", "status": "refunded"},
    },
)

cfg = {"configurable": {"thread_id": "thread-1"}}
paused = graph.invoke({"user_query": "Please refund order #8492"}, config=cfg)
# Pauses at human_approval interrupt()
resumed = graph.invoke(Command(resume={"approved": True}), config=cfg)
print(resumed["final_response"])
```

---

## Project Structure

```text
src/langgraph_blocks/
├── schema.py          # Canonical JSON AST, camelCase/snake_case normalization, safety validation
├── compiler_sg.py     # AST -> StateGraph Python code emitter (TypedDict + nodes + conditional edges)
├── compiler_func.py   # AST -> Functional API (@entrypoint + @task) Python code emitter
├── runtime.py         # Live build_state_graph() builder + WorkflowRunner HITL simulator
├── server.py          # Local HTTP server for Studio UI and /api/compile, /api/simulate, /api/resume
├── cli.py             # CLI entrypoints (studio, init, validate, compile, run)
├── presets.py         # Built-in starter workflow templates
└── static/index.html  # Self-contained Scratch-style web Studio UI
examples/
├── support_refund_hitl.json
└── research_critique_loop.json
tests/
```

---

## Alpha Status & Roadmap

This is an early alpha (`v0.1.0-alpha.1`) validating the structured-block IR and dual LangGraph compilation pipeline.

Planned next:
- Drag-and-drop block reordering and inline variable pill autocomplete in the Studio UI
- Sub-workflow block (`StateGraph` subgraph compilation)
- `LLM Router` multi-branch block and `Try / Fallback` resilience container
- Eval runner integration for testing block workflows across datasets
