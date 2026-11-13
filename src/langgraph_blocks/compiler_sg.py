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


def _check_guardrail(val: Any, rule_type: str, rule_arg: str) -> bool:
    """Evaluate deterministic safety and validation guardrails."""
    text = "" if val is None else str(val)
    if rule_type == "not_empty":
        return bool(text.strip())
    if rule_type == "no_pii":
        has_email = bool(re.search(r"[\\w.+-]+@[\\w-]+\\.[a-zA-Z]{2,}", text))
        has_ssn = bool(re.search(r"\\b\\d{3}-\\d{2}-\\d{4}\\b", text))
        return not (has_email or has_ssn)
    if rule_type == "max_length":
        try:
            return len(text) <= int(rule_arg or 500)
        except ValueError:
            return True
    if rule_type == "regex":
        return bool(re.search(rule_arg or ".*", text))
    return True


def _parse_extracted_json(raw: Any, fields_spec: str) -> dict[str, Any]:
    """Parse JSON from model text or build a typed stub dict for schema fields."""
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    match = re.search(r"\\{[\\s\\S]*\\}", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    fields = [part.split(":")[0].strip() for part in fields_spec.split(",") if part.strip()]
    return {f: text for f in fields} if fields else {"raw": text}


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
    counter = 0

    def next_node_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{sanitize_identifier(prefix, 'node')}_{counter}"

    def build_region(blocks: list[BlockNode], builder_var: str) -> tuple[list[str], list[str]]:
        """Lower a block sequence into (builder_statements, registered_node_names)."""
        region_nodes: list[str] = []
        region_edges: list[str] = []

        def reg_node(prefix: str, body_lines: list[str]) -> str:
            n_name = next_node_id(prefix)
            region_nodes.append(n_name)
            node_defs.append("\n".join([f"    def {n_name}(state: AgentState) -> dict[str, Any]:", *body_lines]))
            return n_name

        def lower_seq(seq: list[BlockNode]) -> tuple[str, str] | None:
            if not seq:
                return None
            first_entry: str | None = None
            prev_exit: str | None = None
            for blk in seq:
                entry, exit_node = lower_blk(blk)
                if first_entry is None:
                    first_entry = entry
                if prev_exit is not None:
                    region_edges.append(f'    {builder_var}.add_edge("{prev_exit}", "{entry}")')
                prev_exit = exit_node
            assert first_entry is not None and prev_exit is not None
            return first_entry, prev_exit

        def lower_blk(blk: BlockNode) -> tuple[str, str]:
            cfg = blk.config

            if blk.type == "ask_llm":
                out_var = sanitize_identifier(str(cfg.get("output_var", "llm_output")))
                model = str(cfg.get("model", "gpt-4.1-mini"))
                prompt = str(cfg.get("prompt", ""))
                n = reg_node(
                    str(cfg.get("name", "ask_llm")),
                    [
                        f"        prompt = _render({prompt!r}, state)",
                        f"        result = _call_llm(llm, prompt, state, {str(cfg.get('name', 'ask_llm'))!r}, {model!r})",
                        f"        return {{{out_var!r}: result}}",
                    ],
                )
                return n, n

            if blk.type == "extract_json":
                out_var = sanitize_identifier(str(cfg.get("output_var", "extracted")))
                src_var = sanitize_identifier(str(cfg.get("source_var", wf.trigger.input_var)))
                fields_spec = str(cfg.get("schema_fields", "summary: str"))
                model = str(cfg.get("model", "gpt-4.1-mini"))
                n = reg_node(
                    str(cfg.get("name", "extract_json")),
                    [
                        f"        prompt = 'Extract JSON fields (' + {fields_spec!r} + ') from: ' + str(state.get({src_var!r}, ''))",
                        f"        raw = _call_llm(llm, prompt, state, 'extract_json', {model!r})",
                        f"        return {{{out_var!r}: _parse_extracted_json(raw, {fields_spec!r})}}",
                    ],
                )
                return n, n

            if blk.type == "agent_loop":
                out_var = sanitize_identifier(str(cfg.get("output_var", "agent_draft")))
                tools_raw = [t.strip() for t in str(cfg.get("tools", "")).split(",") if t.strip()]
                max_steps = max(1, int(cfg.get("max_steps", 5)))
                prompt = str(cfg.get("prompt", ""))
                model = str(cfg.get("model", "claude-sonnet"))
                n = reg_node(
                    str(cfg.get("name", "agent_loop")),
                    [
                        f"        prompt = _render({prompt!r}, state)",
                        "        if react_runner is not None:",
                        f"            result = react_runner(prompt=prompt, tools={tools_raw!r}, max_steps={max_steps}, state=state)",
                        "        else:",
                        f"            result = _call_llm(llm, prompt, state, 'agent_loop', {model!r})",
                        f"        return {{{out_var!r}: result}}",
                    ],
                )
                return n, n

            if blk.type == "rag_retrieve":
                collection = str(cfg.get("collection", "default_kb"))
                query_expr = str(cfg.get("query_expr", "{{user_query}}"))
                top_k = max(1, int(cfg.get("top_k", 3)))
                out_var = sanitize_identifier(str(cfg.get("output_var", "retrieved_docs")))
                n = reg_node(
                    "rag_retrieve",
                    [
                        f"        query = _render({query_expr!r}, state)",
                        "        fn = tool_map.get('rag_retrieve')",
                        "        if fn is not None:",
                        f"            docs = fn({{'collection': {collection!r}, 'query': query, 'top_k': {top_k}}}, state)",
                        "        else:",
                        f"            docs = [f'[{collection} #{{i+1}}] match for {{query}}' for i in range({top_k})]",
                        f"        return {{{out_var!r}: docs}}",
                    ],
                )
                return n, n

            if blk.type == "set_state":
                target_var = sanitize_identifier(str(cfg.get("target_var", "status")))
                value_expr = str(cfg.get("value_expr", ""))
                n = reg_node(
                    "set_state",
                    [f"        return {{{target_var!r}: _render({value_expr!r}, state)}}"],
                )
                return n, n

            if blk.type == "append_memory":
                role = str(cfg.get("role", "assistant"))
                content_expr = str(cfg.get("content_expr", "{{final_response}}"))
                target_var = sanitize_identifier(str(cfg.get("target_var", "messages")))
                if target_var == "messages":
                    n = reg_node(
                        "append_memory",
                        [
                            f"        text = _render({content_expr!r}, state)",
                            f"        return {{'messages': [{{'role': {role!r}, 'content': text}}]}}",
                        ],
                    )
                else:
                    n = reg_node(
                        "append_memory",
                        [
                            f"        text = _render({content_expr!r}, state)",
                            f"        cur = list(state.get({target_var!r}) or [])",
                            f"        return {{{target_var!r}: cur + [{{'role': {role!r}, 'content': text}}]}}",
                        ],
                    )
                return n, n

            if blk.type == "call_tool":
                tool_name = str(cfg.get("tool_name", "tool"))
                arg_expr = str(cfg.get("arg_expr", ""))
                out_var = sanitize_identifier(str(cfg.get("output_var", "tool_result")))
                n = reg_node(
                    tool_name,
                    [
                        f"        arg_val = _render({arg_expr!r}, state)",
                        f"        fn = tool_map.get({tool_name!r})",
                        "        if fn is not None:",
                        "            result = fn.invoke(arg_val) if hasattr(fn, 'invoke') else fn(arg_val, state)",
                        "        else:",
                        f"            result = f'[mock_tool:{tool_name}] {{arg_val}}'",
                        f"        return {{{out_var!r}: result}}",
                    ],
                )
                return n, n

            if blk.type == "http_request":
                method = str(cfg.get("method", "GET")).upper()
                url_expr = str(cfg.get("url_expr", "https://api.example.com/status"))
                body_expr = str(cfg.get("body_expr", ""))
                out_var = sanitize_identifier(str(cfg.get("output_var", "http_response")))
                n = reg_node(
                    "http_request",
                    [
                        f"        url = _render({url_expr!r}, state)",
                        f"        payload = _render({body_expr!r}, state)",
                        "        fn = tool_map.get('http_request')",
                        "        if fn is not None:",
                        f"            result = fn({{'method': {method!r}, 'url': url, 'body': payload}}, state)",
                        "        else:",
                        f"            result = {{'status': 200, 'method': {method!r}, 'url': url, 'body': payload}}",
                        f"        return {{{out_var!r}: result}}",
                    ],
                )
                return n, n

            if blk.type == "sub_workflow":
                wf_name = str(cfg.get("workflow_name", "sub_agent"))
                input_expr = str(cfg.get("input_expr", "{{user_query}}"))
                out_var = sanitize_identifier(str(cfg.get("output_var", "sub_result")))
                n = reg_node(
                    "sub_workflow",
                    [
                        f"        sub_in = _render({input_expr!r}, state)",
                        f"        fn = tool_map.get('workflow:{wf_name}')",
                        "        if fn is not None:",
                        "            result = fn.invoke(sub_in) if hasattr(fn, 'invoke') else fn(sub_in, state)",
                        "        else:",
                        f"            result = f'[sub_workflow:{wf_name}] {{sub_in}}'",
                        f"        return {{{out_var!r}: result}}",
                    ],
                )
                return n, n

            if blk.type == "respond":
                out_var = sanitize_identifier(str(cfg.get("output_var", "final_response")))
                template = str(cfg.get("template", ""))
                n = reg_node(
                    "respond",
                    [f"        return {{{out_var!r}: _render({template!r}, state)}}"],
                )
                return n, n

            if blk.type == "if_else":
                gate = reg_node("if_gate", ["        return {}"])
                join = reg_node("if_join", ["        return {}"])
                left_var = sanitize_identifier(str(cfg.get("left_var", "intent")))
                op = str(cfg.get("operator", "=="))
                right_val = cfg.get("right_val", "")

                then_res = lower_seq(blk.branches.get("then", []))
                else_res = lower_seq(blk.branches.get("else", []))
                region_edges.append(
                    "\n".join(
                        [
                            f"    {builder_var}.add_conditional_edges(",
                            f'        "{gate}",',
                            f"        lambda s: 'true' if _eval_cond(s.get({left_var!r}), {op!r}, {right_val!r}) else 'false',",
                            f'        {{"true": "{then_res[0] if then_res else join}", "false": "{else_res[0] if else_res else join}"}},',
                            "    )",
                        ]
                    )
                )
                if then_res:
                    region_edges.append(f'    {builder_var}.add_edge("{then_res[1]}", "{join}")')
                if else_res:
                    region_edges.append(f'    {builder_var}.add_edge("{else_res[1]}", "{join}")')
                return gate, join

            if blk.type == "llm_router":
                model = str(cfg.get("model", "gpt-4.1-mini"))
                prompt = str(cfg.get("prompt", "Route {{user_query}}"))
                label_a = str(cfg.get("label_a", "billing"))
                label_b = str(cfg.get("label_b", "technical"))
                out_var = sanitize_identifier(str(cfg.get("output_var", "route_choice")))
                gate = reg_node(
                    "llm_router",
                    [
                        f"        prompt = _render({prompt!r}, state)",
                        f"        raw = str(_call_llm(llm, prompt, state, 'llm_router', {model!r})).strip().lower()",
                        f"        if {label_a.lower()!r} in raw:",
                        f"            choice = {label_a!r}",
                        f"        elif {label_b.lower()!r} in raw:",
                        f"            choice = {label_b!r}",
                        "        else:",
                        "            choice = 'fallback'",
                        f"        return {{{out_var!r}: choice}}",
                    ],
                )
                join = reg_node("router_join", ["        return {}"])
                a_res = lower_seq(blk.branches.get("route_a", []))
                b_res = lower_seq(blk.branches.get("route_b", []))
                fb_res = lower_seq(blk.branches.get("fallback", []))
                region_edges.append(
                    "\n".join(
                        [
                            f"    {builder_var}.add_conditional_edges(",
                            f'        "{gate}",',
                            f"        lambda s: 'a' if s.get({out_var!r}) == {label_a!r} else ('b' if s.get({out_var!r}) == {label_b!r} else 'fallback'),",
                            f'        {{"a": "{a_res[0] if a_res else join}", "b": "{b_res[0] if b_res else join}", "fallback": "{fb_res[0] if fb_res else join}"}},',
                            "    )",
                        ]
                    )
                )
                for res in (a_res, b_res, fb_res):
                    if res:
                        region_edges.append(f'    {builder_var}.add_edge("{res[1]}", "{join}")')
                return gate, join

            if blk.type == "guardrail":
                target_var = sanitize_identifier(str(cfg.get("target_var", wf.trigger.input_var)))
                rule_type = str(cfg.get("rule_type", "not_empty"))
                rule_arg = str(cfg.get("rule_arg", ""))
                out_var = sanitize_identifier(str(cfg.get("output_var", "guardrail_passed")))
                gate = reg_node(
                    "guardrail",
                    [
                        f"        passed = _check_guardrail(state.get({target_var!r}), {rule_type!r}, {rule_arg!r})",
                        f"        return {{{out_var!r}: passed}}",
                    ],
                )
                join = reg_node("guard_join", ["        return {}"])
                pass_res = lower_seq(blk.branches.get("pass", []))
                blk_res = lower_seq(blk.branches.get("blocked", []))
                region_edges.append(
                    "\n".join(
                        [
                            f"    {builder_var}.add_conditional_edges(",
                            f'        "{gate}",',
                            f"        lambda s: 'pass' if s.get({out_var!r}) else 'blocked',",
                            f'        {{"pass": "{pass_res[0] if pass_res else join}", "blocked": "{blk_res[0] if blk_res else join}"}},',
                            "    )",
                        ]
                    )
                )
                if pass_res:
                    region_edges.append(f'    {builder_var}.add_edge("{pass_res[1]}", "{join}")')
                if blk_res:
                    region_edges.append(f'    {builder_var}.add_edge("{blk_res[1]}", "{join}")')
                return gate, join

            if blk.type == "human_approval":
                question = str(cfg.get("question", "Approve action?"))
                out_var = sanitize_identifier(str(cfg.get("output_var", "approval")))
                gate = reg_node(
                    "human_gate",
                    [
                        f"        prompt = _render({question!r}, state)",
                        "        decision = interrupt({'question': prompt, 'state_preview': dict(state)})",
                        "        approved = bool(decision.get('approved')) if isinstance(decision, dict) else bool(decision)",
                        f"        return {{{out_var!r}: approved}}",
                    ],
                )
                join = reg_node("hitl_join", ["        return {}"])
                app_res = lower_seq(blk.branches.get("approved", []))
                rej_res = lower_seq(blk.branches.get("rejected", []))
                region_edges.append(
                    "\n".join(
                        [
                            f"    {builder_var}.add_conditional_edges(",
                            f'        "{gate}",',
                            f"        lambda s: 'approved' if s.get({out_var!r}) else 'rejected',",
                            f'        {{"approved": "{app_res[0] if app_res else join}", "rejected": "{rej_res[0] if rej_res else join}"}},',
                            "    )",
                        ]
                    )
                )
                if app_res:
                    region_edges.append(f'    {builder_var}.add_edge("{app_res[1]}", "{join}")')
                if rej_res:
                    region_edges.append(f'    {builder_var}.add_edge("{rej_res[1]}", "{join}")')
                return gate, join

            if blk.type == "repeat_until":
                loop_key = sanitize_identifier(f"_loop_{blk.id}_count")
                cond_var = sanitize_identifier(str(cfg.get("condition_var", "quality_score")))
                op = str(cfg.get("operator", ">="))
                target_val = cfg.get("target_val", "8")
                max_iters = max(1, int(cfg.get("max_iterations", 3)))
                guard = reg_node(
                    "loop_guard",
                    [f"        return {{{loop_key!r}: int(state.get({loop_key!r}, 0)) + 1}}"],
                )
                done = reg_node("loop_done", ["        return {}"])
                body_res = lower_seq(blk.branches.get("body", []))
                if body_res:
                    region_edges.append(f'    {builder_var}.add_edge("{body_res[1]}", "{guard}")')
                    region_edges.append(
                        "\n".join(
                            [
                                f"    {builder_var}.add_conditional_edges(",
                                f'        "{guard}",',
                                f"        lambda s: 'done' if ((int(s.get({loop_key!r}, 0)) > 1 and _eval_cond(s.get({cond_var!r}), {op!r}, {target_val!r})) or int(s.get({loop_key!r}, 0)) > {max_iters}) else 'again',",
                                f'        {{"again": "{body_res[0]}", "done": "{done}"}},',
                                "    )",
                            ]
                        )
                    )
                else:
                    region_edges.append(f'    {builder_var}.add_edge("{guard}", "{done}")')
                return guard, done

            if blk.type == "for_each":
                list_var = sanitize_identifier(str(cfg.get("list_var", "items")))
                item_var = sanitize_identifier(str(cfg.get("item_var", "current_item")))
                idx_key = sanitize_identifier(f"_for_{blk.id}_idx")
                max_items = max(1, int(cfg.get("max_items", 10)))
                guard = reg_node(
                    "foreach_guard",
                    [
                        f"        items = list(state.get({list_var!r}) or [])[:{max_items}]",
                        f"        idx = int(state.get({idx_key!r}, 0))",
                        "        if idx >= len(items):",
                        f"            return {{{idx_key!r}: -1}}",
                        f"        return {{{idx_key!r}: idx + 1, {item_var!r}: items[idx]}}",
                    ],
                )
                done = reg_node("foreach_done", [f"        return {{{idx_key!r}: 0}}"])
                body_res = lower_seq(blk.branches.get("body", []))
                if body_res:
                    region_edges.append(f'    {builder_var}.add_edge("{body_res[1]}", "{guard}")')
                    region_edges.append(
                        "\n".join(
                            [
                                f"    {builder_var}.add_conditional_edges(",
                                f'        "{guard}",',
                                f"        lambda s: 'done' if int(s.get({idx_key!r}, -1)) == -1 else 'next',",
                                f'        {{"next": "{body_res[0]}", "done": "{done}"}},',
                                "    )",
                            ]
                        )
                    )
                else:
                    region_edges.append(f'    {builder_var}.add_edge("{guard}", "{done}")')
                return guard, done

            if blk.type == "parallel":
                split = reg_node("fan_out", ["        return {}"])
                join = reg_node("fan_in", ["        return {}"])
                lane_a = lower_seq(blk.branches.get("laneA", []))
                lane_b = lower_seq(blk.branches.get("laneB", []))
                active = [lane for lane in (lane_a, lane_b) if lane is not None]
                if not active:
                    region_edges.append(f'    {builder_var}.add_edge("{split}", "{join}")')
                elif len(active) == 1:
                    region_edges.append(f'    {builder_var}.add_edge("{split}", "{active[0][0]}")')
                    region_edges.append(f'    {builder_var}.add_edge("{active[0][1]}", "{join}")')
                else:
                    for entry, _ in active:
                        region_edges.append(f'    {builder_var}.add_edge("{split}", "{entry}")')
                    exits_literal = ", ".join(f'"{ex}"' for _, ex in active)
                    region_edges.append(f'    {builder_var}.add_edge([{exits_literal}], "{join}")')
                return split, join

            if blk.type == "try_fallback":
                err_var = sanitize_identifier(str(cfg.get("error_var", "last_error")))
                sub_builder = next_node_id("try_sub")
                sub_stmts, _sub_nodes = build_region(blk.branches.get("try", []), sub_builder)
                sub_compiled = f"{sub_builder}_compiled"
                node_defs.append(
                    "\n".join(
                        [
                            f"    {sub_builder} = StateGraph(AgentState)",
                            *sub_stmts,
                            f"    {sub_compiled} = {sub_builder}.compile()",
                        ]
                    )
                )
                try_exec = reg_node(
                    "try_exec",
                    [
                        "        try:",
                        f"            out = dict({sub_compiled}.invoke(state))",
                        "            if 'messages' in out:",
                        "                out['messages'] = list(out['messages'])[len(state.get('messages') or []):]",
                        f"            out[{err_var!r}] = ''",
                        "            return out",
                        "        except Exception as exc:",
                        "            if exc.__class__.__name__ == 'GraphInterrupt':",
                        "                raise",
                        f"            return {{{err_var!r}: str(exc)}}",
                    ],
                )
                join = reg_node("try_join", ["        return {}"])
                fb_res = lower_seq(blk.branches.get("fallback", []))
                region_edges.append(
                    "\n".join(
                        [
                            f"    {builder_var}.add_conditional_edges(",
                            f'        "{try_exec}",',
                            f"        lambda s: 'fallback' if s.get({err_var!r}) else 'ok',",
                            f'        {{"ok": "{join}", "fallback": "{fb_res[0] if fb_res else join}"}},',
                            "    )",
                        ]
                    )
                )
                if fb_res:
                    region_edges.append(f'    {builder_var}.add_edge("{fb_res[1]}", "{join}")')
                return try_exec, join

            raise ValueError(f"Unsupported block type: {blk.type}")

        seq_res = lower_seq(blocks)
        if seq_res is None:
            noop = reg_node("noop", ["        return {}"])
            region_edges.append(f'    {builder_var}.add_edge(START, "{noop}")')
            region_edges.append(f'    {builder_var}.add_edge("{noop}", END)')
        else:
            region_edges.insert(0, f'    {builder_var}.add_edge(START, "{seq_res[0]}")')
            region_edges.append(f'    {builder_var}.add_edge("{seq_res[1]}", END)')

        stmts = [f'    {builder_var}.add_node("{n}", {n})' for n in region_nodes]
        stmts.extend(region_edges)
        return stmts, region_nodes

    root_stmts, _ = build_region(wf.blocks, "builder")

    state_lines = [
        "class AgentState(TypedDict, total=False):",
        "    messages: Annotated[list[Any], add_messages]",
    ]
    for key in state_vars:
        py_type = "int" if (key.startswith("_loop_") or key.startswith("_for_")) else "Any"
        state_lines.append(f"    {key}: {py_type}")

    lines: list[str] = [
        f'"""Generated by langgraph-blocks (StateGraph target) for workflow: {wf.trigger.name}."""',
        "",
        "import json",
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
        *root_stmts,
        "",
        f"    saver = checkpointer if checkpointer is not None else {checkpointer_cls}()",
        "    return builder.compile(checkpointer=saver)",
        "",
        "",
        'if __name__ == "__main__":',
        f"    {fn_name} = build_workflow()",
        f'    print({fn_name}.invoke({{{sanitize_identifier(wf.trigger.input_var)!r}: "Hello"}}, config={{"configurable": {{"thread_id": "demo"}}}}))',
        "",
    ]
    return "\n".join(lines)
