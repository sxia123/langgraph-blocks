"""Tests for Functional API (@entrypoint/@task) codegen and real LangGraph execution."""

import ast
import uuid

from langgraph.types import Command

from langgraph_blocks import compile_to_functional_api
from langgraph_blocks.presets import get_builtin_presets


def _load_functional_builder(code: str):
    ast.parse(code)
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns["build_workflow"]


def test_functional_codegen_parses_empty_and_presets() -> None:
    empty_code = compile_to_functional_api({"trigger": {"name": "empty"}, "blocks": []})
    ast.parse(empty_code)

    for raw in get_builtin_presets().values():
        code = compile_to_functional_api(raw)
        ast.parse(code)


def test_generated_functional_api_executes_hitl_on_real_langgraph() -> None:
    presets = get_builtin_presets()
    code = compile_to_functional_api(presets["support_refund_hitl"])
    build_workflow = _load_functional_builder(code)

    def mock_llm(prompt: str, state: dict) -> str:
        return "refund" if "refund" in str(state.get("user_query", "")).lower() else "general"

    app = build_workflow(
        llm=mock_llm,
        tools={
            "lookup_order": lambda arg, _s: f"ORD-8492({arg})",
            "issue_refund": lambda arg, _s: f"REFUNDED-{arg}",
        },
    )

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    paused = app.invoke({"user_query": "Please refund order 8492"}, config=config)
    assert "__interrupt__" in paused

    resumed = app.invoke(Command(resume={"approved": False}), config=config)
    assert "declined" in resumed["final_response"].lower()


def test_generated_functional_api_executes_parallel_and_loop() -> None:
    presets = get_builtin_presets()
    code = compile_to_functional_api(presets["research_critique_loop"])
    build_workflow = _load_functional_builder(code)

    attempts = {"score_calls": 0}

    def mock_llm(prompt: str, state: dict):
        if "Score" in prompt:
            attempts["score_calls"] += 1
            return 5 if attempts["score_calls"] == 1 else 8
        return f"Brief({state.get('web_notes')}, {state.get('internal_notes')})"

    app = build_workflow(
        llm=mock_llm,
        tools={
            "web_search": lambda q, _s: f"WEB:{q}",
            "search_kb": lambda q, _s: f"KB:{q}",
        },
    )
    out = app.invoke({"user_query": "Blocks"}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    assert attempts["score_calls"] == 2
    assert "WEB:Blocks" in out["final_response"]
    assert "KB:Blocks" in out["final_response"]
