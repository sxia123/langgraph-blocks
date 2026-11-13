"""Tests for StateGraph Python code generation and real LangGraph execution."""

import ast
import uuid

from langgraph.types import Command

from langgraph_blocks import compile_to_state_graph
from langgraph_blocks.presets import get_builtin_presets


def _load_compiled_graph(code: str):
    ast.parse(code)
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns["build_workflow"]


def test_stategraph_codegen_parses_empty_and_presets() -> None:
    empty_code = compile_to_state_graph({"trigger": {"name": "empty"}, "blocks": []})
    ast.parse(empty_code)

    for raw in get_builtin_presets().values():
        code = compile_to_state_graph(raw)
        ast.parse(code)


def test_generated_stategraph_executes_hitl_refund_on_real_langgraph() -> None:
    presets = get_builtin_presets()
    code = compile_to_state_graph(presets["support_refund_hitl"])
    build_workflow = _load_compiled_graph(code)

    def mock_llm(prompt: str, state: dict) -> str:
        return "refund" if "refund" in str(state.get("user_query", "")).lower() else "general"

    graph = build_workflow(
        llm=mock_llm,
        tools={
            "lookup_order": lambda arg, _s: f"ORD-8492({arg})",
            "issue_refund": lambda arg, _s: f"REFUNDED-{arg}",
        },
    )

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    paused = graph.invoke({"user_query": "Please refund order 8492"}, config=config)
    assert "__interrupt__" in paused

    resumed = graph.invoke(Command(resume={"approved": True}), config=config)
    assert "REFUNDED-ORD-8492" in resumed["final_response"]


def test_generated_stategraph_executes_parallel_and_bounded_loop() -> None:
    presets = get_builtin_presets()
    code = compile_to_state_graph(presets["research_critique_loop"])
    build_workflow = _load_compiled_graph(code)

    attempts = {"score_calls": 0}

    def mock_llm(prompt: str, state: dict):
        if "Score" in prompt:
            attempts["score_calls"] += 1
            return 6 if attempts["score_calls"] == 1 else 9
        return f"Brief using {state.get('web_notes')} + {state.get('internal_notes')}"

    graph = build_workflow(
        llm=mock_llm,
        tools={
            "web_search": lambda q, _s: f"WEB({q})",
            "search_kb": lambda q, _s: f"KB({q})",
        },
    )
    out = graph.invoke({"user_query": "LangGraph"}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    assert attempts["score_calls"] == 2
    assert "WEB(LangGraph)" in out["final_response"]
    assert "KB(LangGraph)" in out["final_response"]
