"""Compiler from WorkflowSchema AST to LangGraph Functional API (@entrypoint / @task)."""

from __future__ import annotations

from typing import Any

from .compiler_sg import _RUNTIME_HELPERS, _checkpointer_import, _llm_call_kwargs
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
                model = str(cfg.get("model", "") or "")
                prompt = str(cfg.get("prompt", ""))
                llm_kw = _llm_call_kwargs(cfg)
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> Any:",
                            f"        prompt = _render({prompt!r}, cur_state)",
                            f"        return _call_llm(llm, prompt, cur_state, {t_name!r}, {model!r}{llm_kw})",
                        ]
                    )
                )
                out.append(f"{pad}state[{out_var!r}] = {t_name}(dict(state)).result()")

            elif blk.type == "extract_json":
                t_name = next_task_name(str(cfg.get("name", "extract_json")))
                out_var = sanitize_identifier(str(cfg.get("output_var", "extracted")))
                src_var = sanitize_identifier(str(cfg.get("source_var", wf.trigger.input_var)))
                fields_spec = str(cfg.get("schema_fields", "summary: str"))
                model = str(cfg.get("model", "") or "")
                llm_kw = _llm_call_kwargs(cfg)
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> dict[str, Any]:",
                            f"        prompt = 'Extract JSON fields (' + {fields_spec!r} + ') from: ' + str(cur_state.get({src_var!r}, ''))",
                            f"        raw = _call_llm(llm, prompt, cur_state, 'extract_json', {model!r}{llm_kw})",
                            f"        return _parse_extracted_json(raw, {fields_spec!r})",
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
                model = str(cfg.get("model", "") or "")
                llm_kw = _llm_call_kwargs(cfg)
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> Any:",
                            f"        prompt = _render({prompt!r}, cur_state)",
                            "        if react_runner is not None:",
                            f"            return react_runner(prompt=prompt, tools={tools_raw!r}, max_steps={max_steps}, state=cur_state)",
                            f"        return _call_llm(llm, prompt, cur_state, {t_name!r}, {model!r}{llm_kw})",
                        ]
                    )
                )
                out.append(f"{pad}state[{out_var!r}] = {t_name}(dict(state)).result()")

            elif blk.type == "rag_retrieve":
                t_name = next_task_name("rag_retrieve")
                collection = str(cfg.get("collection", "default_kb"))
                query_expr = str(cfg.get("query_expr", "{{user_query}}"))
                top_k = max(1, int(cfg.get("top_k", 3)))
                out_var = sanitize_identifier(str(cfg.get("output_var", "retrieved_docs")))
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> list[Any]:",
                            f"        query = _render({query_expr!r}, cur_state)",
                            "        fn = tool_map.get('rag_retrieve')",
                            "        if fn is not None:",
                            f"            return fn({{'collection': {collection!r}, 'query': query, 'top_k': {top_k}}}, cur_state)",
                            f"        return [f'[{collection} #{{i+1}}] match for {{query}}' for i in range({top_k})]",
                        ]
                    )
                )
                out.append(f"{pad}state[{out_var!r}] = {t_name}(dict(state)).result()")

            elif blk.type == "set_state":
                target_var = sanitize_identifier(str(cfg.get("target_var", "status")))
                value_expr = str(cfg.get("value_expr", ""))
                out.append(f"{pad}state[{target_var!r}] = _render({value_expr!r}, state)")

            elif blk.type == "append_memory":
                role = str(cfg.get("role", "assistant"))
                content_expr = str(cfg.get("content_expr", "{{final_response}}"))
                target_var = sanitize_identifier(str(cfg.get("target_var", "messages")))
                out.append(f"{pad}_msg_text = _render({content_expr!r}, state)")
                out.append(
                    f"{pad}state[{target_var!r}] = list(state.get({target_var!r}) or []) + [{{'role': {role!r}, 'content': _msg_text}}]"
                )

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

            elif blk.type == "http_request":
                t_name = next_task_name("http_request")
                method = str(cfg.get("method", "GET")).upper()
                url_expr = str(cfg.get("url_expr", "https://api.example.com/status"))
                body_expr = str(cfg.get("body_expr", ""))
                out_var = sanitize_identifier(str(cfg.get("output_var", "http_response")))
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> Any:",
                            f"        url = _render({url_expr!r}, cur_state)",
                            f"        payload = _render({body_expr!r}, cur_state)",
                            "        fn = tool_map.get('http_request')",
                            "        if fn is not None:",
                            f"            return fn({{'method': {method!r}, 'url': url, 'body': payload}}, cur_state)",
                            f"        return {{'status': 200, 'method': {method!r}, 'url': url, 'body': payload}}",
                        ]
                    )
                )
                out.append(f"{pad}state[{out_var!r}] = {t_name}(dict(state)).result()")

            elif blk.type == "sub_workflow":
                t_name = next_task_name("sub_workflow")
                wf_name = str(cfg.get("workflow_name", "sub_agent"))
                input_expr = str(cfg.get("input_expr", "{{user_query}}"))
                out_var = sanitize_identifier(str(cfg.get("output_var", "sub_result")))
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> Any:",
                            f"        sub_in = _render({input_expr!r}, cur_state)",
                            f"        fn = tool_map.get('workflow:{wf_name}')",
                            "        if fn is not None:",
                            "            return fn.invoke(sub_in) if hasattr(fn, 'invoke') else fn(sub_in, cur_state)",
                            f"        return f'[sub_workflow:{wf_name}] {{sub_in}}'",
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

            elif blk.type == "llm_router":
                t_name = next_task_name("llm_router")
                model = str(cfg.get("model", "") or "")
                prompt = str(cfg.get("prompt", "Route {{user_query}}"))
                label_a = str(cfg.get("label_a", "billing"))
                label_b = str(cfg.get("label_b", "technical"))
                out_var = sanitize_identifier(str(cfg.get("output_var", "route_choice")))
                llm_kw = _llm_call_kwargs(cfg)
                task_defs.append(
                    "\n".join(
                        [
                            "    @task",
                            f"    def {t_name}(cur_state: dict[str, Any]) -> str:",
                            f"        prompt = _render({prompt!r}, cur_state)",
                            f"        raw = str(_call_llm(llm, prompt, cur_state, 'llm_router', {model!r}{llm_kw})).strip().lower()",
                            f"        if {label_a.lower()!r} in raw:",
                            f"            return {label_a!r}",
                            f"        if {label_b.lower()!r} in raw:",
                            f"            return {label_b!r}",
                            "        return 'fallback'",
                        ]
                    )
                )
                out.append(f"{pad}state[{out_var!r}] = {t_name}(dict(state)).result()")
                out.append(f"{pad}if state[{out_var!r}] == {label_a!r}:")
                out.extend(emit_sequence(blk.branches.get("route_a", []), indent + 4))
                out.append(f"{pad}elif state[{out_var!r}] == {label_b!r}:")
                out.extend(emit_sequence(blk.branches.get("route_b", []), indent + 4))
                out.append(f"{pad}else:")
                out.extend(emit_sequence(blk.branches.get("fallback", []), indent + 4))

            elif blk.type == "guardrail":
                target_var = sanitize_identifier(str(cfg.get("target_var", wf.trigger.input_var)))
                rule_type = str(cfg.get("rule_type", "not_empty"))
                rule_arg = str(cfg.get("rule_arg", ""))
                out_var = sanitize_identifier(str(cfg.get("output_var", "guardrail_passed")))
                out.append(
                    f"{pad}state[{out_var!r}] = _check_guardrail(state.get({target_var!r}), {rule_type!r}, {rule_arg!r})"
                )
                out.append(f"{pad}if state[{out_var!r}]:")
                out.extend(emit_sequence(blk.branches.get("pass", []), indent + 4))
                out.append(f"{pad}else:")
                out.extend(emit_sequence(blk.branches.get("blocked", []), indent + 4))

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

            elif blk.type == "for_each":
                list_var = sanitize_identifier(str(cfg.get("list_var", "items")))
                item_var = sanitize_identifier(str(cfg.get("item_var", "current_item")))
                max_items = max(1, int(cfg.get("max_items", 10)))
                out.append(f"{pad}for _item in list(state.get({list_var!r}) or [])[:{max_items}]:")
                out.append(f"{pad}    state[{item_var!r}] = _item")
                out.extend(emit_sequence(blk.branches.get("body", []), indent + 4))

            elif blk.type == "try_fallback":
                err_var = sanitize_identifier(str(cfg.get("error_var", "last_error")))
                out.append(f"{pad}try:")
                out.extend(emit_sequence(blk.branches.get("try", []), indent + 4))
                out.append(f"{pad}except Exception as _exc:")
                out.append(f"{pad}    if _exc.__class__.__name__ == 'GraphInterrupt':")
                out.append(f"{pad}        raise")
                out.append(f"{pad}    state[{err_var!r}] = str(_exc)")
                out.extend(emit_sequence(blk.branches.get("fallback", []), indent + 4))

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
        "import json",
        "import os",
        "import re",
        "import urllib.error",
        "import urllib.parse",
        "import urllib.request",
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
