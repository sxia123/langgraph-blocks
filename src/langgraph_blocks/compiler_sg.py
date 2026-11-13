"""Compiler from WorkflowSchema AST to idiomatic LangGraph StateGraph Python code."""

from __future__ import annotations

from typing import Any

from .schema import BlockNode, WorkflowSchema, collect_state_vars, parse_workflow, sanitize_identifier

_RUNTIME_HELPERS = '''
def _render(template: str, state: dict[str, Any]) -> str:
    """Safely interpolate {{var}} placeholders from state."""
    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        val = state.get(key, "")
        return "" if val is None else str(val)
    return re.sub(r"\\{\\{\\s*([a-zA-Z0-9_]+)\\s*\\}\\}", repl, template)


def _eval_cond(actual: Any, op: str, expected: Any) -> bool:
    """Evaluate a structured branch or loop condition safely."""
    if op == "in":
        return str(expected) in str(actual or "")
    if op in (">=", ">", "<=", "<"):
        try:
            a_num = float(actual if actual not in (None, "") else 0)
            b_num = float(expected)
            if op == ">=":
                return a_num >= b_num
            if op == ">":
                return a_num > b_num
            if op == "<=":
                return a_num <= b_num
            return a_num < b_num
        except (TypeError, ValueError):
            pass
    a_str = "" if actual is None else str(actual)
    b_str = "" if expected is None else str(expected)
    return a_str != b_str if op == "!=" else a_str == b_str


def _call_llm(llm_fn: Any, prompt: str, state: dict[str, Any], step_name: str, model: str) -> Any:
    if llm_fn is None:
        return f"[{model}:{step_name}] {prompt[:120]}"
    if hasattr(llm_fn, "invoke"):
        res = llm_fn.invoke(prompt)
        return getattr(res, "content", res)
    return llm_fn(prompt, state)
'''.strip()


def _checkpointer_import(checkpointer: str) -> str:
    if checkpointer == "PostgresSaver":
        return "from langgraph.checkpoint.postgres import PostgresSaver"
    if checkpointer == "SqliteSaver":
        return "from langgraph.checkpoint.sqlite import SqliteSaver"
    return "from langgraph.checkpoint.memory import MemorySaver"


def compile_to_state_graph(workflow: WorkflowSchema | dict[str, Any]) -> str:
    """Lower a structured block AST into standalone runnable StateGraph Python code."""
    wf = workflow if isinstance(workflow, WorkflowSchema) else parse_workflow(workflow)
    state_vars = collect_state_vars(wf)
    checkpointer_cls = wf.trigger.checkpointer or "MemorySaver"
    fn_name = sanitize_identifier(wf.trigger.name, "agent_workflow")

    node_defs: list[str] = []
    node_names: list[str] = []
    edges: list[str] = []
    counter = 0

    def next_node(prefix: str) -> str:
        nonlocal counter
        counter += 1
        name = f"{sanitize_identifier(prefix, 'node')}_{counter}"
        node_names.append(name)
        return name

    def lower_sequence(blocks: list[BlockNode]) -> tuple[str, str] | None:
        if not blocks:
            return None
        first_entry: str | None = None
        prev_exit: str | None = None
        for blk in blocks:
            entry, exit_node = lower_block(blk)
            if first_entry is None:
                first_entry = entry
            if prev_exit is not None:
                edges.append(f'    builder.add_edge("{prev_exit}", "{entry}")')
            prev_exit = exit_node
        assert first_entry is not None and prev_exit is not None
        return first_entry, prev_exit

    def lower_block(blk: BlockNode) -> tuple[str, str]:
        cfg = blk.config

        if blk.type == "ask_llm":
            n_name = next_node(cfg.get("name", "ask_llm"))
            out_var = sanitize_identifier(str(cfg.get("output_var", "llm_output")))
            model = str(cfg.get("model", "gpt-4.1-mini"))
            prompt = str(cfg.get("prompt", ""))
            node_defs.append(
                "\n".join(
                    [
                        f"    def {n_name}(state: AgentState) -> dict[str, Any]:",
                        f"        prompt = _render({prompt!r}, state)",
                        f"        result = _call_llm(llm, prompt, state, {n_name!r}, {model!r})",
                        f"        return {{{out_var!r}: result}}",
                    ]
                )
            )
            return n_name, n_name

        if blk.type == "agent_loop":
            n_name = next_node(cfg.get("name", "agent_loop"))
            out_var = sanitize_identifier(str(cfg.get("output_var", "agent_draft")))
            tools_raw = [t.strip() for t in str(cfg.get("tools", "")).split(",") if t.strip()]
            max_steps = max(1, int(cfg.get("max_steps", 5)))
            prompt = str(cfg.get("prompt", ""))
            model = str(cfg.get("model", "claude-sonnet"))
            node_defs.append(
                "\n".join(
                    [
                        f"    def {n_name}(state: AgentState) -> dict[str, Any]:",
                        f"        prompt = _render({prompt!r}, state)",
                        "        if react_runner is not None:",
                        f"            result = react_runner(prompt=prompt, tools={tools_raw!r}, max_steps={max_steps}, state=state)",
                        "        else:",
                        f"            result = _call_llm(llm, prompt, state, {n_name!r}, {model!r})",
                        f"        return {{{out_var!r}: result}}",
                    ]
                )
            )
            return n_name, n_name

        if blk.type == "call_tool":
            n_name = next_node(cfg.get("tool_name", "call_tool"))
            tool_name = str(cfg.get("tool_name", "tool"))
            arg_expr = str(cfg.get("arg_expr", ""))
            out_var = sanitize_identifier(str(cfg.get("output_var", "tool_result")))
            node_defs.append(
                "\n".join(
                    [
                        f"    def {n_name}(state: AgentState) -> dict[str, Any]:",
                        f"        arg_val = _render({arg_expr!r}, state)",
                        f"        fn = tool_map.get({tool_name!r})",
                        "        if fn is not None:",
                        "            result = fn.invoke(arg_val) if hasattr(fn, 'invoke') else fn(arg_val, state)",
                        "        else:",
                        f"            result = f'[mock_tool:{tool_name}] {{arg_val}}'",
                        f"        return {{{out_var!r}: result}}",
                    ]
                )
            )
            return n_name, n_name

        if blk.type == "respond":
            n_name = next_node("respond")
            out_var = sanitize_identifier(str(cfg.get("output_var", "final_response")))
            template = str(cfg.get("template", ""))
            node_defs.append(
                "\n".join(
                    [
                        f"    def {n_name}(state: AgentState) -> dict[str, Any]:",
                        f"        return {{{out_var!r}: _render({template!r}, state)}}",
                    ]
                )
            )
            return n_name, n_name

        if blk.type == "if_else":
            gate = next_node("if_gate")
            join = next_node("if_join")
            left_var = sanitize_identifier(str(cfg.get("left_var", "intent")))
            op = str(cfg.get("operator", "=="))
            right_val = cfg.get("right_val", "")

            node_defs.append(f"    def {gate}(state: AgentState) -> dict[str, Any]:\n        return {{}}")
            node_defs.append(f"    def {join}(state: AgentState) -> dict[str, Any]:\n        return {{}}")

            then_res = lower_sequence(blk.branches.get("then", []))
            else_res = lower_sequence(blk.branches.get("else", []))

            true_target = then_res[0] if then_res else join
            false_target = else_res[0] if else_res else join

            edges.append(
                "\n".join(
                    [
                        "    builder.add_conditional_edges(",
                        f'        "{gate}",',
                        f"        lambda s: 'true' if _eval_cond(s.get({left_var!r}), {op!r}, {right_val!r}) else 'false',",
                        f'        {{"true": "{true_target}", "false": "{false_target}"}},',
                        "    )",
                    ]
                )
            )
            if then_res:
                edges.append(f'    builder.add_edge("{then_res[1]}", "{join}")')
            if else_res:
                edges.append(f'    builder.add_edge("{else_res[1]}", "{join}")')
            return gate, join

        if blk.type == "human_approval":
            gate = next_node("human_gate")
            join = next_node("hitl_join")
            question = str(cfg.get("question", "Approve action?"))
            out_var = sanitize_identifier(str(cfg.get("output_var", "approval")))

            node_defs.append(
                "\n".join(
                    [
                        f"    def {gate}(state: AgentState) -> dict[str, Any]:",
                        f"        prompt = _render({question!r}, state)",
                        "        decision = interrupt({'question': prompt, 'state_preview': dict(state)})",
                        "        approved = bool(decision.get('approved')) if isinstance(decision, dict) else bool(decision)",
                        f"        return {{{out_var!r}: approved}}",
                    ]
                )
            )
            node_defs.append(f"    def {join}(state: AgentState) -> dict[str, Any]:\n        return {{}}")

            app_res = lower_sequence(blk.branches.get("approved", []))
            rej_res = lower_sequence(blk.branches.get("rejected", []))
            app_target = app_res[0] if app_res else join
            rej_target = rej_res[0] if rej_res else join

            edges.append(
                "\n".join(
                    [
                        "    builder.add_conditional_edges(",
                        f'        "{gate}",',
                        f"        lambda s: 'approved' if s.get({out_var!r}) else 'rejected',",
                        f'        {{"approved": "{app_target}", "rejected": "{rej_target}"}},',
                        "    )",
                    ]
                )
            )
            if app_res:
                edges.append(f'    builder.add_edge("{app_res[1]}", "{join}")')
            if rej_res:
                edges.append(f'    builder.add_edge("{rej_res[1]}", "{join}")')
            return gate, join

        if blk.type == "repeat_until":
            guard = next_node("loop_guard")
            done = next_node("loop_done")
            loop_key = sanitize_identifier(f"_loop_{blk.id}_count")
            cond_var = sanitize_identifier(str(cfg.get("condition_var", "quality_score")))
            op = str(cfg.get("operator", ">="))
            target_val = cfg.get("target_val", "8")
            max_iters = max(1, int(cfg.get("max_iterations", 3)))

            node_defs.append(
                "\n".join(
                    [
                        f"    def {guard}(state: AgentState) -> dict[str, Any]:",
                        f"        return {{{loop_key!r}: int(state.get({loop_key!r}, 0)) + 1}}",
                    ]
                )
            )
            node_defs.append(f"    def {done}(state: AgentState) -> dict[str, Any]:\n        return {{}}")

            body_res = lower_sequence(blk.branches.get("body", []))
            if body_res:
                edges.append(f'    builder.add_edge("{body_res[1]}", "{guard}")')
                edges.append(
                    "\n".join(
                        [
                            "    builder.add_conditional_edges(",
                            f'        "{guard}",',
                            f"        lambda s: 'done' if ((int(s.get({loop_key!r}, 0)) > 1 and _eval_cond(s.get({cond_var!r}), {op!r}, {target_val!r})) or int(s.get({loop_key!r}, 0)) > {max_iters}) else 'again',",
                            f'        {{"again": "{body_res[0]}", "done": "{done}"}},',
                            "    )",
                        ]
                    )
                )
            else:
                edges.append(f'    builder.add_edge("{guard}", "{done}")')
            return guard, done

        if blk.type == "parallel":
            split = next_node("fan_out")
            join = next_node("fan_in")
            node_defs.append(f"    def {split}(state: AgentState) -> dict[str, Any]:\n        return {{}}")
            node_defs.append(f"    def {join}(state: AgentState) -> dict[str, Any]:\n        return {{}}")

            lane_a = lower_sequence(blk.branches.get("laneA", []))
            lane_b = lower_sequence(blk.branches.get("laneB", []))
            active = [lane for lane in (lane_a, lane_b) if lane is not None]
            if not active:
                edges.append(f'    builder.add_edge("{split}", "{join}")')
            elif len(active) == 1:
                edges.append(f'    builder.add_edge("{split}", "{active[0][0]}")')
                edges.append(f'    builder.add_edge("{active[0][1]}", "{join}")')
            else:
                for entry, _ in active:
                    edges.append(f'    builder.add_edge("{split}", "{entry}")')
                exits_literal = ", ".join(f'"{ex}"' for _, ex in active)
                edges.append(f'    builder.add_edge([{exits_literal}], "{join}")')
            return split, join

        raise ValueError(f"Unsupported block type: {blk.type}")

    seq_res = lower_sequence(wf.blocks)
    if seq_res is None:
        noop = next_node("noop")
        node_defs.append(f"    def {noop}(state: AgentState) -> dict[str, Any]:\n        return {{}}")
        edges.append(f'    builder.add_edge(START, "{noop}")')
        edges.append(f'    builder.add_edge("{noop}", END)')
    else:
        edges.insert(0, f'    builder.add_edge(START, "{seq_res[0]}")')
        edges.append(f'    builder.add_edge("{seq_res[1]}", END)')

    state_lines = [
        "class AgentState(TypedDict, total=False):",
        "    messages: Annotated[list[Any], add_messages]",
    ]
    for key in state_vars:
        py_type = "int" if key.startswith("_loop_") else "Any"
        state_lines.append(f"    {key}: {py_type}")

    lines: list[str] = [
        f'"""Generated by langgraph-blocks (StateGraph target) for workflow: {wf.trigger.name}."""',
        "",
        "import re",
        "from typing import Annotated, Any, Callable, TypedDict",
        "",
        "from langgraph.graph import END, START, StateGraph, add_messages",
        "from langgraph.types import interrupt",
        _checkpointer_import(checkpointer_cls),
        "",
        "",
        *state_lines,
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
        '    """Build and compile the LangGraph StateGraph workflow."""',
        "    tool_map = tools or {}",
        "",
        *node_defs,
        "",
        "    builder = StateGraph(AgentState)",
    ]

    for n_name in node_names:
        lines.append(f'    builder.add_node("{n_name}", {n_name})')

    lines.append("")
    lines.extend(edges)
    lines.append("")
    lines.append(f"    saver = checkpointer if checkpointer is not None else {checkpointer_cls}()")
    lines.append("    return builder.compile(checkpointer=saver)")
    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append(f"    {fn_name} = build_workflow()")
    lines.append(
        f'    print({fn_name}.invoke({{{sanitize_identifier(wf.trigger.input_var)!r}: "Hello"}}, config={{"configurable": {{"thread_id": "demo"}}}}))'
    )
    lines.append("")
    return "\n".join(lines)
