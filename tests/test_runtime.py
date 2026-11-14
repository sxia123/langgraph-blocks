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


def test_runtime_llm_generation_settings() -> None:
    wf = {
        "trigger": {"name": "ai_runtime_opts", "input_var": "user_query", "checkpointer": "MemorySaver"},
        "blocks": [
            {
                "id": "ask_1",
                "type": "ask_llm",
                "config": {
                    "name": "answer",
                    "model": "gemini-2.5-pro",
                    "temperature": 0.7,
                    "max_tokens": 256,
                    "system_prompt": "System: help with {{user_query}}",
                    "prompt": "User question",
                    "output_var": "reply",
                },
            }
        ],
    }

    seen: list[tuple[str, dict]] = []

    class FakeChat:
        def __init__(self, bound=None):
            self.bound = bound or {}

        def bind(self, **kw):
            return FakeChat({**self.bound, **kw})

        def invoke(self, text: str):
            seen.append((text, self.bound))
            return f"{self.bound.get('temperature')}/{self.bound.get('max_tokens')}|{text}"

    runner = WorkflowRunner()
    res = runner.start(wf, "routing", llm=FakeChat())
    assert res.status == "done"
    assert seen and seen[0][1] == {"temperature": 0.7, "max_tokens": 256}
    assert "System: help with routing" in res.state["reply"]


def test_runtime_custom_endpoint_with_env_and_transient_api_key(monkeypatch) -> None:
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    captured_headers: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured_headers.append(self.headers.get("Authorization", ""))
            raw = json.dumps({"choices": [{"message": {"content": "RUNTIME_EP_OK"}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    try:
        monkeypatch.setenv("CUSTOM_EP_TEST_TOKEN", "env-secret-123")
        wf = {
            "trigger": {"name": "rt_ep", "input_var": "user_query", "checkpointer": "MemorySaver"},
            "blocks": [
                {
                    "id": "a1",
                    "type": "agent_loop",
                    "config": {
                        "name": "agent_ep",
                        "llm_mode": "custom_endpoint",
                        "endpoint_url": f"http://127.0.0.1:{port}/v1",
                        "api_key_env": "CUSTOM_EP_TEST_TOKEN",
                        "model": "mistral-local",
                        "max_steps": 3,
                        "prompt": "{{user_query}}",
                        "output_var": "out_a",
                    },
                }
            ],
        }
        runner = WorkflowRunner()
        res1 = runner.start(wf, "test env key")
        assert res1.status == "done"
        assert res1.state["out_a"] == "RUNTIME_EP_OK"
        assert captured_headers[-1] == "Bearer env-secret-123"

        res2 = runner.start(wf, "test override key", api_key="session-override-456")
        assert res2.state["out_a"] == "RUNTIME_EP_OK"
        assert captured_headers[-1] == "Bearer session-override-456"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=2)


def test_runtime_per_block_forks_and_container_forks() -> None:
    # 1) build_state_graph: parallel fork (with empty pass-through branch) + repeat_until with conditional fork
    wf_sg = {
        "trigger": {"name": "rt_sg_forks", "input_var": "user_query", "checkpointer": "MemorySaver"},
        "blocks": [
            {
                "id": "pstep",
                "type": "set_state",
                "config": {
                    "target_var": "base",
                    "value_expr": "go",
                    "fork_mode": "parallel",
                    "fork_branches": [
                        {"key": "b1", "label": "One"},
                        {"key": "b2", "label": "Empty"},
                    ],
                },
                "branches": {
                    "b1": [{"id": "s1", "type": "set_state", "config": {"target_var": "flag1", "value_expr": "Y"}}],
                    "b2": [],
                },
            },
            {
                "id": "loop1",
                "type": "repeat_until",
                "config": {
                    "condition_var": "score",
                    "operator": ">=",
                    "target_val": "8",
                    "max_iterations": 2,
                    "fork_mode": "conditional",
                    "fork_branches": [
                        {"key": "hit", "label": "Hit", "var": "flag1", "op": "==", "val": "Y", "is_default": False},
                        {"key": "miss", "label": "Miss", "is_default": True},
                    ],
                },
                "branches": {
                    "body": [{"id": "setscore", "type": "set_state", "config": {"target_var": "score", "value_expr": "9"}}],
                    "hit": [{"id": "sh", "type": "set_state", "config": {"target_var": "loop_tag", "value_expr": "OK:{{base}}:{{score}}"}}],
                    "miss": [{"id": "sm", "type": "set_state", "config": {"target_var": "loop_tag", "value_expr": "NO"}}],
                },
            },
        ],
    }
    graph = build_state_graph(wf_sg)
    sg_out = graph.invoke({"user_query": "run"}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    assert sg_out["flag1"] == "Y"
    assert sg_out["loop_tag"] == "OK:go:9"

    # 2) WorkflowRunner: human_approval with conditional fork + pause/resume
    wf_runner = {
        "trigger": {"name": "rt_runner_fork", "input_var": "user_query", "checkpointer": "MemorySaver"},
        "blocks": [
            {
                "id": "ha1",
                "type": "human_approval",
                "config": {
                    "question": "Approve?",
                    "output_var": "approval",
                    "fork_mode": "conditional",
                    "fork_branches": [
                        {"key": "yes", "label": "Approved", "var": "decision", "op": "==", "val": "approved", "is_default": False},
                        {"key": "no", "label": "Denied", "is_default": True},
                    ],
                },
                "branches": {
                    "approved": [{"id": "da", "type": "set_state", "config": {"target_var": "decision", "value_expr": "approved"}}],
                    "rejected": [{"id": "dr", "type": "set_state", "config": {"target_var": "decision", "value_expr": "rejected"}}],
                    "yes": [{"id": "ry", "type": "set_state", "config": {"target_var": "outcome", "value_expr": "APPROVED"}}],
                    "no": [{"id": "rn", "type": "set_state", "config": {"target_var": "outcome", "value_expr": "DENIED"}}],
                },
            }
        ],
    }
    runner = WorkflowRunner()
    paused_yes = runner.start(wf_runner, "please")
    assert paused_yes.status == "paused"
    resumed_yes = runner.resume(paused_yes.resume_token, True)
    assert resumed_yes.status == "done"
    assert resumed_yes.state["outcome"] == "APPROVED"

    paused_no = runner.start(wf_runner, "please")
    assert paused_no.status == "paused"
    resumed_no = runner.resume(paused_no.resume_token, False)
    assert resumed_no.status == "done"
    assert resumed_no.state["outcome"] == "DENIED"

