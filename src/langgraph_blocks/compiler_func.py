"""Compiler from WorkflowSchema AST to LangGraph Functional API (@entrypoint / @task)."""

from __future__ import annotations

from typing import Any

from .compiler_sg import _RUNTIME_HELPERS, _checkpointer_import
from .schema import BlockNode, WorkflowSchema, parse_workflow, sanitize_identifier


def compile_to_functional_api(workflow: WorkflowSchema | dict[str, Any]) -> str:
    """Lower a structured block AST into standalone runnable Functional API Python code."""
    wf = workflow if isinstance(workflow, WorkflowSchema) else parse_workflow(workflow)
    checkpointer_cls = wf.trigger.checkpointer or "MemorySaver"
    fn_name = sanitize_identifier(wf.trigger.name, "agent_workflow")
    input_key = sanitize_identifier(wf.trigger.input_var, "user_query")

    task_defs: list[str] = []
    task_counter = 0

    def next_task_name(prefix: str) -> str:
        nonlocal task_counter
        task_counter += 1
        return f"{sanitize_identifier(prefix, 'step')}_task_{task_counter}"

    def emit_sequence(blocks: list[BlockNode], indent: int) -> list[str]:
        pad = " " * indent
        if not blocks:
            return [f"{pad}pass"]

        out: list[str] = []
        for blk in blocks:
            cfg = blk.config
            if blk.type == "ask_llm":
                t_name = next_task_name(str(cfg.get("name", "ask_llm")))
                out_var = sanitize_identifier(str(cfg.get("output_var", "llm_output")))
                model = str(cfg.get("model", "gpt-4.1-mini"))
                prompt = str(cfg.get("prompt", ""))
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> Any:",
                            f"        prompt = _render({prompt!r}, cur_state)",
                            f"        return _call_llm(llm, prompt, cur_state, {t_name!r}, {model!r})",
                        ]
                    )
                )
                out.append(f"{pad}state[{out_var!r}] = {t_name}(dict(state)).result()")

            elif blk.type == "agent_loop":
                t_name = next_task_name(str(cfg.get("name", "agent_loop")))
                out_var = sanitize_identifier(str(cfg.get("output_var", "agent_draft")))
                tools_raw = [t.strip() for t in str(cfg.get("tools", "")).split(",") if t.strip()]
                max_steps = max(1, int(cfg.get("max_steps", 5)))
                prompt = str(cfg.get("prompt", ""))
                model = str(cfg.get("model", "claude-sonnet"))
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> Any:",
                            f"        prompt = _render({prompt!r}, cur_state)",
                            "        if react_runner is not None:",
                            f"            return react_runner(prompt=prompt, tools={tools_raw!r}, max_steps={max_steps}, state=cur_state)",
                            f"        return _call_llm(llm, prompt, cur_state, {t_name!r}, {model!r})",
                        ]
                    )
                )
                out.append(f"{pad}state[{out_var!r}] = {t_name}(dict(state)).result()")

            elif blk.type == "call_tool":
                t_name = next_task_name(str(cfg.get("tool_name", "call_tool")))
                tool_name = str(cfg.get("tool_name", "tool"))
                arg_expr = str(cfg.get("arg_expr", ""))
                out_var = sanitize_identifier(str(cfg.get("output_var", "tool_result")))
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> Any:",
                            f"        arg_val = _render({arg_expr!r}, cur_state)",
                            f"        fn = tool_map.get({tool_name!r})",
                            "        if fn is not None:",
                            "            return fn.invoke(arg_val) if hasattr(fn, 'invoke') else fn(arg_val, cur_state)",
                            f"        return f'[mock_tool:{tool_name}] {{arg_val}}'",
                        ]
                    )
                )
                out.append(f"{pad}state[{out_var!r}] = {t_name}(dict(state)).result()")

            elif blk.type == "respond":
                out_var = sanitize_identifier(str(cfg.get("output_var", "final_response")))
                template = str(cfg.get("template", ""))
                out.append(f"{pad}state[{out_var!r}] = _render({template!r}, state)")
                out.append(f"{pad}return dict(state)")

            elif blk.type == "if_else":
                left_var = sanitize_identifier(str(cfg.get("left_var", "intent")))
                op = str(cfg.get("operator", "=="))
                right_val = cfg.get("right_val", "")
                out.append(f"{pad}if _eval_cond(state.get({left_var!r}), {op!r}, {right_val!r}):")
                out.extend(emit_sequence(blk.branches.get("then", []), indent + 4))
                out.append(f"{pad}else:")
                out.extend(emit_sequence(blk.branches.get("else", []), indent + 4))

            elif blk.type == "human_approval":
                question = str(cfg.get("question", "Approve action?"))
                out_var = sanitize_identifier(str(cfg.get("output_var", "approval")))
                out.append(
                    f"{pad}_decision = interrupt({{'question': _render({question!r}, state), 'state_preview': dict(state)}})"
                )
                out.append(
                    f"{pad}state[{out_var!r}] = bool(_decision.get('approved')) if isinstance(_decision, dict) else bool(_decision)"
                )
                out.append(f"{pad}if state[{out_var!r}]:")
                out.extend(emit_sequence(blk.branches.get("approved", []), indent + 4))
                out.append(f"{pad}else:")
                out.extend(emit_sequence(blk.branches.get("rejected", []), indent + 4))

            elif blk.type == "repeat_until":
                loop_var = sanitize_identifier(f"_iter_{blk.id}")
                cond_var = sanitize_identifier(str(cfg.get("condition_var", "quality_score")))
                op = str(cfg.get("operator", ">="))
                target_val = cfg.get("target_val", "8")
                max_iters = max(1, int(cfg.get("max_iterations", 3)))
                out.append(f"{pad}for {loop_var} in range({max_iters}):")
                out.extend(emit_sequence(blk.branches.get("body", []), indent + 4))
                out.append(
                    f"{pad}    if _eval_cond(state.get({cond_var!r}), {op!r}, {target_val!r}):"
                )
                out.append(f"{pad}        break")

            elif blk.type == "parallel":
                lane_a = blk.branches.get("laneA", [])
                lane_b = blk.branches.get("laneB", [])
                futures: list[str] = []
                for lane_label, lane_blocks in (("lane_a", lane_a), ("lane_b", lane_b)):
                    if not lane_blocks:
                        continue
                    lane_task = next_task_name(lane_label)
                    lane_body = emit_sequence(lane_blocks, 8)
                    task_defs.append(
                        "\n".join(
                            [
                                "    @task",
                                f"    def {lane_task}(branch_state: dict[str, Any]) -> dict[str, Any]:",
                                "        state = dict(branch_state)",
                                *lane_body,
                                "        return state",
                            ]
                        )
                    )
                    fut_var = f"fut_{lane_task}"
                    out.append(f"{pad}{fut_var} = {lane_task}(dict(state))")
                    futures.append(fut_var)
                for fut_var in futures:
                    out.append(f"{pad}state.update({fut_var}.result())")

        return out

    body_lines = emit_sequence(wf.blocks, 8)
    if not body_lines or not body_lines[-1].strip().startswith("return "):
        body_lines.append("        return dict(state)")

    lines: list[str] = [
        f'"""Generated by langgraph-blocks (Functional API target) for workflow: {wf.trigger.name}."""',
        "",
        "import re",
        "from typing import Any, Callable",
        "",
        "from langgraph.func import entrypoint, task",
        "from langgraph.types import interrupt",
        _checkpointer_import(checkpointer_cls),
        "",
        "",
        _RUNTIME_HELPERS,
        "",
        "",
        "def build_workflow(",
        "    *,",
        "    llm: Any | None = None,",
        "    tools: dict[str, Callable[..., Any]] | None = None,",
        "    react_runner: Callable[..., Any] | None = None,",
        "    checkpointer: Any | None = None,",
        "):",
        '    """Build and compile the LangGraph Functional API workflow."""',
        "    tool_map = tools or {}",
        f"    saver = checkpointer if checkpointer is not None else {checkpointer_cls}()",
        "",
        *[t + "\n" for t in task_defs],
        "    @entrypoint(checkpointer=saver)",
        f"    def {fn_name}(input_data: dict[str, Any], *, previous: dict[str, Any] | None = None) -> dict[str, Any]:",
        "        state: dict[str, Any] = {",
        f"            {input_key!r}: input_data.get({input_key!r}, input_data.get('query', '')),",
        "            '_previous': previous or {},",
        "        }",
        *body_lines,
        "",
        f"    return {fn_name}",
        "",
        "",
        'if __name__ == "__main__":',
        f"    {fn_name} = build_workflow()",
        f'    print({fn_name}.invoke({{{input_key!r}: "Hello"}}, config={{"configurable": {{"thread_id": "demo"}}}}))',
        "",
    ]
    return "\n".join(lines)
