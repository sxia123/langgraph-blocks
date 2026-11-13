"""Command-line interface for LangGraph Blocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .compiler_func import compile_to_functional_api
from .compiler_sg import compile_to_state_graph
from .presets import get_builtin_presets
from .runtime import WorkflowRunner
from .schema import ValidationError, collect_state_vars, parse_workflow
from .server import DEFAULT_STUDIO_PORT, serve_studio


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langgraph-blocks",
        description="Structured Scratch-style block workflows compiled to LangGraph.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create a starter workflow JSON file.")
    p_init.add_argument("output", type=Path, help="Destination JSON file path.")
    p_init.add_argument(
        "--preset",
        choices=sorted(get_builtin_presets().keys()),
        default="support_refund_hitl",
        help="Starter preset template.",
    )

    p_val = sub.add_parser("validate", help="Validate a workflow JSON file.")
    p_val.add_argument("workflow", type=Path, help="Path to workflow JSON.")

    p_comp = sub.add_parser("compile", help="Compile workflow JSON into Python LangGraph code.")
    p_comp.add_argument("workflow", type=Path, help="Path to workflow JSON.")
    p_comp.add_argument(
        "--target",
        choices=("stategraph", "functional"),
        default="stategraph",
        help="LangGraph compilation target.",
    )
    p_comp.add_argument("-o", "--output", type=Path, default=None, help="Optional output .py file.")

    p_run = sub.add_parser("run", help="Simulate executing a workflow JSON locally.")
    p_run.add_argument("workflow", type=Path, help="Path to workflow JSON.")
    p_run.add_argument("--input", default="Please refund order #8492", help="Input query string.")
    p_run.add_argument(
        "--approve",
        action="store_true",
        help="Automatically approve if execution pauses at a human_approval interrupt.",
    )

    p_studio = sub.add_parser("studio", help="Launch the local web-based Blocks Studio UI.")
    p_studio.add_argument("--host", default="127.0.0.1", help="Bind host.")
    p_studio.add_argument("--port", type=int, default=DEFAULT_STUDIO_PORT, help="Bind port.")
    p_studio.add_argument("--no-browser", action="store_true", help="Do not auto-open browser.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        presets = get_builtin_presets()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(presets[args.preset], indent=2) + "\n", encoding="utf-8")
        print(f"Created starter workflow ({args.preset}) at {args.output}")
        return 0

    if args.command == "validate":
        try:
            wf = parse_workflow(_load_json(args.workflow))
        except (OSError, ValueError, ValidationError) as exc:
            print(f"Validation failed: {exc}", file=sys.stderr)
            return 1
        print(f"Valid workflow '{wf.trigger.name}'. State keys: {', '.join(collect_state_vars(wf))}")
        return 0

    if args.command == "compile":
        try:
            wf = parse_workflow(_load_json(args.workflow))
            code = (
                compile_to_state_graph(wf)
                if args.target == "stategraph"
                else compile_to_functional_api(wf)
            )
        except (OSError, ValueError, ValidationError) as exc:
            print(f"Compilation failed: {exc}", file=sys.stderr)
            return 1
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(code, encoding="utf-8")
            print(f"Compiled ({args.target}) -> {args.output}")
        else:
            print(code)
        return 0

    if args.command == "run":
        try:
            wf = parse_workflow(_load_json(args.workflow))
        except (OSError, ValueError, ValidationError) as exc:
            print(f"Error loading workflow: {exc}", file=sys.stderr)
            return 1
        runner = WorkflowRunner()
        res = runner.start(wf, args.input)
        while res.status == "paused" and res.resume_token:
            res = runner.resume(res.resume_token, bool(args.approve))
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    if args.command == "studio":
        serve_studio(host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
