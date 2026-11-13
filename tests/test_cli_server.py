"""Tests for CLI commands and the local Studio HTTP server."""

import json
import threading
import urllib.request
from pathlib import Path

from langgraph_blocks import cli
from langgraph_blocks.server import create_studio_server


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
            assert "LangGraph Blocks Studio" in resp.read().decode("utf-8")

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
