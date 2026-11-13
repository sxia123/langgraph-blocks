"""Tests for CLI commands and the local Studio HTTP server."""

import json
import threading
import urllib.request
from pathlib import Path

from langgraph_blocks import cli
from langgraph_blocks.server import (
    DEFAULT_STUDIO_PORT,
    RESERVED_PORTS,
    create_studio_server,
    create_studio_server_with_fallback,
)


def test_cli_init_validate_compile_and_run(tmp_path: Path, capsys) -> None:
    wf_path = tmp_path / "flow.json"
    py_path = tmp_path / "flow_sg.py"

    assert cli.main(["init", str(wf_path), "--preset", "support_refund_hitl"]) == 0
    assert wf_path.is_file()

    assert cli.main(["validate", str(wf_path)]) == 0
    out = capsys.readouterr().out
    assert "Valid workflow" in out

    assert cli.main(["compile", str(wf_path), "--target", "stategraph", "-o", str(py_path)]) == 0
    assert "StateGraph" in py_path.read_text(encoding="utf-8")
    capsys.readouterr()

    assert cli.main(["run", str(wf_path), "--input", "Refund #8492", "--approve"]) == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["status"] == "done"


def test_studio_http_server_endpoints() -> None:
    server = create_studio_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/") as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "LangGraph Blocks Studio" in html
            assert "MODEL_GROUPS" not in html
            assert "custom_endpoint" in html

        with urllib.request.urlopen(f"{base}/api/examples") as resp:
            examples = json.loads(resp.read().decode("utf-8"))["examples"]
            assert "support_refund_hitl" in examples

        req = urllib.request.Request(
            f"{base}/api/compile",
            data=json.dumps({"workflow": examples["support_refund_hitl"], "target": "functional"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            compiled = json.loads(resp.read().decode("utf-8"))
            assert "@entrypoint" in compiled["code"]

        sim_req = urllib.request.Request(
            f"{base}/api/simulate",
            data=json.dumps({"workflow": examples["support_refund_hitl"], "input": "Refund order"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(sim_req) as resp:
            sim_res = json.loads(resp.read().decode("utf-8"))
            assert sim_res["status"] == "paused"
            token = sim_res["resume_token"]

        res_req = urllib.request.Request(
            f"{base}/api/resume",
            data=json.dumps({"resume_token": token, "approved": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(res_req) as resp:
            done = json.loads(resp.read().decode("utf-8"))
            assert done["status"] == "done"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_studio_server_fallback_skips_busy_and_reserved_ports() -> None:
    assert DEFAULT_STUDIO_PORT not in RESERVED_PORTS
    assert 8000 in RESERVED_PORTS
    assert 8765 in RESERVED_PORTS

    blocker = create_studio_server(host="127.0.0.1", port=0)
    busy_port = blocker.server_address[1]
    blocker_thread = threading.Thread(target=blocker.serve_forever, daemon=True)
    blocker_thread.start()

    try:
        fallback = create_studio_server_with_fallback(host="127.0.0.1", port=busy_port)
        try:
            chosen_port = fallback.server_address[1]
            assert chosen_port != busy_port
            assert chosen_port not in RESERVED_PORTS
        finally:
            fallback.server_close()

        skip_reserved = create_studio_server_with_fallback(host="127.0.0.1", port=8000)
        try:
            assert skip_reserved.server_address[1] not in RESERVED_PORTS
        finally:
            skip_reserved.server_close()
    finally:
        blocker.shutdown()
        blocker.server_close()
        blocker_thread.join(timeout=2)


def test_studio_simulate_passes_transient_api_key() -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen_auth: list[str] = []

    class EpHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen_auth.append(self.headers.get("Authorization", ""))
            raw = json.dumps({"choices": [{"message": {"content": "SIM_KEY_OK"}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):
            pass

    ep_srv = HTTPServer(("127.0.0.1", 0), EpHandler)
    ep_port = ep_srv.server_address[1]
    ep_thread = threading.Thread(target=ep_srv.serve_forever, daemon=True)
    ep_thread.start()

    studio_srv = create_studio_server(host="127.0.0.1", port=0)
    studio_port = studio_srv.server_address[1]
    studio_thread = threading.Thread(target=studio_srv.serve_forever, daemon=True)
    studio_thread.start()

    try:
        wf = {
            "trigger": {"name": "studio_ep", "input_var": "user_query", "checkpointer": "MemorySaver"},
            "blocks": [
                {
                    "id": "a1",
                    "type": "ask_llm",
                    "config": {
                        "name": "ask",
                        "llm_mode": "custom_endpoint",
                        "endpoint_url": f"http://127.0.0.1:{ep_port}/v1",
                        "model": "ep-model",
                        "prompt": "{{user_query}}",
                        "output_var": "ans",
                    },
                }
            ],
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{studio_port}/api/simulate",
            data=json.dumps({"workflow": wf, "input": "ping", "api_key": "transient-token"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            assert out["status"] == "done"
            assert out["state"]["ans"] == "SIM_KEY_OK"
            assert seen_auth == ["Bearer transient-token"]
    finally:
        studio_srv.shutdown()
        studio_srv.server_close()
        studio_thread.join(timeout=2)
        ep_srv.shutdown()
        ep_srv.server_close()
        ep_thread.join(timeout=2)
