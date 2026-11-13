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


def test_generated_functional_api_executes_expanded_blocks() -> None:
    wf = {
        "trigger": {"name": "expanded_fn", "input_var": "user_query", "checkpointer": "MemorySaver"},
        "blocks": [
            {
                "id": "g1",
                "type": "guardrail",
                "config": {"target_var": "user_query", "rule_type": "not_empty", "output_var": "non_empty"},
                "branches": {
                    "pass": [
                        {"id": "ej1", "type": "extract_json", "config": {"source_var": "user_query", "schema_fields": "kind: str", "output_var": "parsed"}},
                        {"id": "rag1", "type": "rag_retrieve", "config": {"collection": "faq", "query_expr": "{{user_query}}", "top_k": 2, "output_var": "hits"}},
                        {
                            "id": "rt1",
                            "type": "llm_router",
                            "config": {"prompt": "Route {{user_query}}", "label_a": "billing", "label_b": "tech", "output_var": "route"},
                            "branches": {
                                "route_a": [],
                                "route_b": [{"id": "http1", "type": "http_request", "config": {"method": "GET", "url_expr": "https://api.test/ok", "output_var": "http_res"}}],
                                "fallback": [],
                            },
                        },
                        {
                            "id": "fe1",
                            "type": "for_each",
                            "config": {"list_var": "hits", "item_var": "hit", "max_items": 3},
                            "branches": {
                                "body": [{"id": "mem1", "type": "append_memory", "config": {"role": "assistant", "content_expr": "{{hit}}", "target_var": "messages"}}]
                            },
                        },
                        {
                            "id": "tf1",
                            "type": "try_fallback",
                            "config": {"error_var": "err"},
                            "branches": {
                                "try": [{"id": "c1", "type": "call_tool", "config": {"tool_name": "boom", "arg_expr": "1", "output_var": "res"}}],
                                "fallback": [{"id": "sub1", "type": "sub_workflow", "config": {"workflow_name": "recover", "input_expr": "{{err}}", "output_var": "sub_out"}}],
                            },
                        },
                        {"id": "s1", "type": "set_state", "config": {"target_var": "done_flag", "value_expr": "yes"}},
                        {"id": "resp1", "type": "respond", "config": {"template": "{{sub_out}}:{{done_flag}}", "output_var": "final_response"}},
                    ],
                    "blocked": [{"id": "resp2", "type": "respond", "config": {"template": "EMPTY", "output_var": "final_response"}}],
                },
            }
        ],
    }
    code = compile_to_functional_api(wf)
    build_workflow = _load_functional_builder(code)

    def mock_llm(prompt: str, _s: dict) -> str:
        if "Extract JSON" in prompt:
            return '{"kind": "tech"}'
        if "Route" in prompt:
            return "tech"
        return "ok"

    def boom(_a, _s):
        raise ValueError("bad_call")

    app = build_workflow(
        llm=mock_llm,
        tools={
            "boom": boom,
            "workflow:recover": lambda err_in, _s: f"RECOVERED({err_in})",
        },
    )
    out = app.invoke({"user_query": "API error 500"}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    assert out["parsed"] == {"kind": "tech"}
    assert len(out["messages"]) == 2
    assert out["http_res"]["status"] == 200
    assert out["final_response"] == "RECOVERED(bad_call):yes"


def test_functional_api_calls_custom_endpoint_live() -> None:
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            received.append({"path": self.path, "body": body})
            resp = {"choices": [{"message": {"content": "ENDPOINT_FN_OK"}}]}
            raw = json.dumps(resp).encode("utf-8")
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
        wf = {
            "trigger": {"name": "custom_ep_fn", "input_var": "user_query", "checkpointer": "MemorySaver"},
            "blocks": [
                {
                    "id": "ask_fn",
                    "type": "ask_llm",
                    "config": {
                        "name": "fn_step",
                        "llm_mode": "custom_endpoint",
                        "endpoint_url": f"http://127.0.0.1:{port}/v1/chat/completions",
                        "model": "qwen-local",
                        "prompt": "Hi {{user_query}}",
                        "output_var": "answer",
                    },
                }
            ],
        }
        code = compile_to_functional_api(wf)
        build_workflow = _load_functional_builder(code)
        app = build_workflow(llm=None)
        out = app.invoke({"user_query": "world"}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
        assert out["answer"] == "ENDPOINT_FN_OK"
        assert received[0]["path"] == "/v1/chat/completions"
        assert received[0]["body"]["model"] == "qwen-local"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=2)
