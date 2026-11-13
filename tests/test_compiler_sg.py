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


def test_generated_stategraph_executes_expanded_blocks_on_real_langgraph() -> None:
    wf = {
        "trigger": {"name": "expanded_sg", "input_var": "user_query", "checkpointer": "MemorySaver"},
        "blocks": [
            {
                "id": "g1",
                "type": "guardrail",
                "config": {"target_var": "user_query", "rule_type": "no_pii", "output_var": "guard_ok"},
                "branches": {
                    "pass": [
                        {
                            "id": "ej1",
                            "type": "extract_json",
                            "config": {"source_var": "user_query", "schema_fields": "category: str", "output_var": "parsed"},
                        },
                        {
                            "id": "rag1",
                            "type": "rag_retrieve",
                            "config": {"collection": "kb", "query_expr": "{{user_query}}", "top_k": 2, "output_var": "docs"},
                        },
                        {
                            "id": "rt1",
                            "type": "llm_router",
                            "config": {"prompt": "Route {{user_query}}", "label_a": "billing", "label_b": "tech", "output_var": "route"},
                            "branches": {
                                "route_a": [
                                    {"id": "sub1", "type": "sub_workflow", "config": {"workflow_name": "billing_flow", "input_expr": "{{user_query}}", "output_var": "routed_res"}}
                                ],
                                "route_b": [],
                                "fallback": [
                                    {"id": "set1", "type": "set_state", "config": {"target_var": "routed_res", "value_expr": "fallback"}}
                                ],
                            },
                        },
                        {
                            "id": "fe1",
                            "type": "for_each",
                            "config": {"list_var": "docs", "item_var": "doc_item", "max_items": 2},
                            "branches": {
                                "body": [
                                    {"id": "mem1", "type": "append_memory", "config": {"role": "assistant", "content_expr": "{{doc_item}}", "target_var": "audit_log"}}
                                ]
                            },
                        },
                        {
                            "id": "http1",
                            "type": "http_request",
                            "config": {"method": "POST", "url_expr": "https://api.example.com/notify", "body_expr": "{{routed_res}}", "output_var": "http_res"},
                        },
                        {
                            "id": "tf1",
                            "type": "try_fallback",
                            "config": {"error_var": "err"},
                            "branches": {
                                "try": [
                                    {"id": "t_fail", "type": "call_tool", "config": {"tool_name": "flaky", "arg_expr": "x", "output_var": "flaky_res"}}
                                ],
                                "fallback": [
                                    {"id": "fb_set", "type": "set_state", "config": {"target_var": "recovery", "value_expr": "recovered:{{err}}"}}
                                ],
                            },
                        },
                        {"id": "resp_ok", "type": "respond", "config": {"template": "{{routed_res}}|{{recovery}}", "output_var": "final_response"}},
                    ],
                    "blocked": [
                        {"id": "resp_blk", "type": "respond", "config": {"template": "BLOCKED_PII", "output_var": "final_response"}}
                    ],
                },
            }
        ],
    }
    code = compile_to_state_graph(wf)
    build_workflow = _load_compiled_graph(code)

    def mock_llm(prompt: str, state: dict) -> str:
        if "Extract JSON" in prompt:
            return '{"category": "invoice"}'
        if "Route" in prompt:
            return "billing" if "invoice" in str(state.get("user_query", "")).lower() else "other"
        return "ok"

    def flaky(_arg, _s):
        raise RuntimeError("upstream_timeout")

    graph = build_workflow(
        llm=mock_llm,
        tools={
            "rag_retrieve": lambda cfg, _s: [f"doc1:{cfg['query']}", f"doc2:{cfg['collection']}"],
            "workflow:billing_flow": lambda inp, _s: f"SUB({inp})",
            "http_request": lambda req, _s: {"status": 201, "method": req["method"], "body": req["body"]},
            "flaky": flaky,
        },
    )

    blocked_out = graph.invoke({"user_query": "email me at alice@example.com"}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    assert blocked_out["guard_ok"] is False
    assert blocked_out["final_response"] == "BLOCKED_PII"

    ok_out = graph.invoke({"user_query": "Need my invoice copy"}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    assert ok_out["guard_ok"] is True
    assert ok_out["parsed"] == {"category": "invoice"}
    assert len(ok_out["audit_log"]) == 2
    assert ok_out["http_res"]["status"] == 201
    assert ok_out["err"] == "upstream_timeout"
    assert "SUB(Need my invoice copy)|recovered:upstream_timeout" == ok_out["final_response"]
