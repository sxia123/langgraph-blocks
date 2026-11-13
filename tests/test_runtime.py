"""Tests for in-memory build_state_graph and WorkflowRunner."""

import uuid

from langgraph.types import Command

from langgraph_blocks import WorkflowRunner, build_state_graph
from langgraph_blocks.presets import get_builtin_presets


def test_in_memory_build_state_graph_runs_on_real_langgraph() -> None:
    presets = get_builtin_presets()
    graph = build_state_graph(presets["support_refund_hitl"])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # General branch (no interrupt)
    general_res = graph.invoke({"user_query": "What is your return window?"}, config=config)
    assert "__interrupt__" not in general_res
    assert general_res["intent"] == "general"
    assert general_res["final_response"]

    # Refund branch (hits real interrupt -> resume)
    refund_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    paused = graph.invoke({"user_query": "Please refund order #8492"}, config=refund_cfg)
    assert "__interrupt__" in paused
    done = graph.invoke(Command(resume={"approved": True}), config=refund_cfg)
    assert "RF-991" in done["final_response"]


def test_workflow_runner_pause_resume_and_loop() -> None:
    presets = get_builtin_presets()
    runner = WorkflowRunner()

    res = runner.start(presets["support_refund_hitl"], "Refund order #8492")
    assert res.status == "paused"
    assert res.resume_token is not None
    assert "Approve refund" in (res.interrupt_payload or {}).get("question", "")

    final_res = runner.resume(res.resume_token, True)
    assert final_res.status == "done"
    assert "RF-991" in final_res.state["final_response"]

    loop_res = runner.start(presets["research_critique_loop"], "LangGraph")
    assert loop_res.status == "done"
    assert loop_res.state["quality_score"] >= 8
    assert loop_res.state["final_response"]


def test_runtime_graph_and_runner_support_expanded_blocks() -> None:
    wf = {
        "trigger": {"name": "expanded_runtime", "input_var": "user_query", "checkpointer": "MemorySaver"},
        "blocks": [
            {
                "id": "g1",
                "type": "guardrail",
                "config": {"target_var": "user_query", "rule_type": "no_pii", "output_var": "safe"},
                "branches": {
                    "pass": [
                        {"id": "ej1", "type": "extract_json", "config": {"source_var": "user_query", "schema_fields": "intent: str", "output_var": "parsed"}},
                        {"id": "rag1", "type": "rag_retrieve", "config": {"collection": "kb", "query_expr": "{{user_query}}", "top_k": 2, "output_var": "docs"}},
                        {
                            "id": "rt1",
                            "type": "llm_router",
                            "config": {"prompt": "Route {{user_query}}", "label_a": "billing", "label_b": "tech", "output_var": "route"},
                            "branches": {
                                "route_a": [{"id": "s_a", "type": "set_state", "config": {"target_var": "dept", "value_expr": "BILLING"}}],
                                "route_b": [{"id": "s_b", "type": "set_state", "config": {"target_var": "dept", "value_expr": "TECH"}}],
                                "fallback": [{"id": "s_f", "type": "set_state", "config": {"target_var": "dept", "value_expr": "GENERAL"}}],
                            },
                        },
                        {
                            "id": "fe1",
                            "type": "for_each",
                            "config": {"list_var": "docs", "item_var": "cur", "max_items": 2},
                            "branches": {
                                "body": [{"id": "mem1", "type": "append_memory", "config": {"role": "assistant", "content_expr": "{{cur}}", "target_var": "notes"}}]
                            },
                        },
                        {"id": "http1", "type": "http_request", "config": {"method": "GET", "url_expr": "https://api.example.com/{{dept}}", "output_var": "http_res"}},
                        {
                            "id": "tf1",
                            "type": "try_fallback",
                            "config": {"error_var": "err"},
                            "branches": {
                                "try": [{"id": "bad_call", "type": "call_tool", "config": {"tool_name": "fail_tool", "arg_expr": "x", "output_var": "x_res"}}],
                                "fallback": [{"id": "sub1", "type": "sub_workflow", "config": {"workflow_name": "backup", "input_expr": "{{err}}", "output_var": "sub_res"}}],
                            },
                        },
                        {"id": "resp_ok", "type": "respond", "config": {"template": "{{dept}}|{{sub_res}}", "output_var": "final_response"}},
                    ],
                    "blocked": [
                        {"id": "resp_blk", "type": "respond", "config": {"template": "PII_BLOCKED", "output_var": "final_response"}}
                    ],
                },
            }
        ],
    }

    def mock_llm(prompt: str, _s: dict) -> str:
        if "Extract JSON" in prompt:
            return '{"intent": "billing"}'
        if "Route" in prompt:
            return "billing"
        return "ok"

    def fail_tool(_arg, _s):
        raise RuntimeError("boom")

    tools = {
        "fail_tool": fail_tool,
        "workflow:backup": lambda inp, _s: f"BACKUP({inp})",
    }

    graph = build_state_graph(wf, llm=mock_llm, tools=tools)
    sg_ok = graph.invoke({"user_query": "Check invoice"}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    assert sg_ok["parsed"] == {"intent": "billing"}
    assert len(sg_ok["notes"]) == 2
    assert sg_ok["http_res"]["url"] == "https://api.example.com/BILLING"
    assert sg_ok["final_response"] == "BILLING|BACKUP(boom)"

    sg_blk = graph.invoke({"user_query": "ssn 123-45-6789"}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    assert sg_blk["final_response"] == "PII_BLOCKED"

    runner = WorkflowRunner()
    run_ok = runner.start(wf, "Check invoice", llm=mock_llm, tools=tools)
    assert run_ok.status == "done"
    assert run_ok.state["parsed"] == {"intent": "billing"}
    assert len(run_ok.state["notes"]) == 2
    assert run_ok.state["final_response"] == "BILLING|BACKUP(boom)"

    run_blk = runner.start(wf, "contact me at bob@test.com", llm=mock_llm, tools=tools)
    assert run_blk.status == "done"
    assert run_blk.state["final_response"] == "PII_BLOCKED"
