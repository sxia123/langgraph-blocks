"""Local HTTP server for the LangGraph Blocks Studio UI and compiler API."""

from __future__ import annotations

import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .compiler_func import compile_to_functional_api
from .compiler_sg import compile_to_state_graph
from .presets import get_builtin_presets
from .runtime import WorkflowRunner
from .schema import ValidationError, collect_state_vars, parse_workflow

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_studio_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer serving the Blocks Studio UI and API."""
    runner = WorkflowRunner()

    class StudioHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON request body must be an object.")
            return data

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                html_path = STATIC_DIR / "index.html"
                if not html_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Studio UI asset not found.")
                    return
                data = html_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if self.path == "/api/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return

            if self.path == "/api/examples":
                self._send_json(HTTPStatus.OK, {"examples": get_builtin_presets()})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._read_json()
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if self.path == "/api/validate":
                raw_wf = body.get("workflow", body)
                try:
                    wf = parse_workflow(raw_wf)
                except ValidationError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"valid": False, "errors": exc.errors})
                    return
                self._send_json(HTTPStatus.OK, {"valid": True, "state_vars": collect_state_vars(wf)})
                return

            if self.path == "/api/compile":
                raw_wf = body.get("workflow", body)
                target = str(body.get("target", "stategraph")).lower()
                try:
                    wf = parse_workflow(raw_wf)
                    if target == "stategraph":
                        code = compile_to_state_graph(wf)
                    elif target in ("functional", "func"):
                        code = compile_to_functional_api(wf)
                    else:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": f"Unsupported target '{target}'. Expected 'stategraph' or 'functional'."},
                        )
                        return
                except ValidationError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc), "errors": exc.errors})
                    return
                self._send_json(
                    HTTPStatus.OK,
                    {"target": target, "code": code, "state_vars": collect_state_vars(wf)},
                )
                return

            if self.path == "/api/simulate":
                raw_wf = body.get("workflow")
                if not isinstance(raw_wf, dict):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'workflow' object."})
                    return
                input_data = body.get("input", "")
                try:
                    result = runner.start(raw_wf, input_data)
                except ValidationError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc), "errors": exc.errors})
                    return
                self._send_json(HTTPStatus.OK, result.to_dict())
                return

            if self.path == "/api/resume":
                token = str(body.get("resume_token", ""))
                approved = bool(body.get("approved", False))
                try:
                    result = runner.resume(token, approved)
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, result.to_dict())
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {self.path}"})

    return ThreadingHTTPServer((host, port), StudioHandler)


def serve_studio(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    """Run the local Blocks Studio server until interrupted."""
    server = create_studio_server(host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"LangGraph Blocks Studio running at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
