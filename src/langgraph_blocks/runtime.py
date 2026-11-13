"""In-memory LangGraph StateGraph runtime builder and step-trace runner."""

from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, TypedDict

from .schema import BlockNode, WorkflowSchema, collect_state_vars, parse_workflow, sanitize_identifier


def render_template(template: str, state: dict[str, Any]) -> str:
    """Safely interpolate {{var}} placeholders from workflow state."""

    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        val = state.get(key, "")
        return "" if val is None else str(val)

    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", repl, str(template or ""))


def eval_condition(actual: Any, op: str, expected: Any) -> bool:
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


def _invoke_llm(llm: Any, prompt: str, state: dict[str, Any], step_name: str, model: str) -> Any:
    if llm is None:
        if "classify" in step_name.lower() or "intent" in step_name.lower():
            q = str(state.get("user_query", "")).lower()
            return "refund" if "refund" in q else "general"
        if "score" in step_name.lower():
            cur = int(state.get("_sim_score", 6) or 6)
            next_score = cur + 2
            state["_sim_score"] = next_score
            return next_score
        return f"[{model}:{step_name}] Generated response for: {prompt[:80]}"
    if hasattr(llm, "invoke"):
        res = llm.invoke(prompt)
        return getattr(res, "content", res)
    return llm(prompt, state)


def _invoke_tool(tool_map: dict[str, Callable[..., Any]], tool_name: str, arg_val: str, state: dict[str, Any]) -> Any:
    fn = tool_map.get(tool_name)
    if fn is not None:
        return fn.invoke(arg_val) if hasattr(fn, "invoke") else fn(arg_val, state)
    if tool_name == "lookup_order":
        return {"order_id": "8492", "amount": "$49.00", "duplicate_charge": True}
    if tool_name == "issue_refund":
        return {"refund_id": "RF-991", "status": "refunded", "amount": "$49.00"}
    return f"[mock_tool:{tool_name}] {arg_val}"


def build_state_graph(
    workflow: WorkflowSchema | dict[str, Any],
    *,
    llm: Any | None = None,
    tools: dict[str, Callable[..., Any]] | None = None,
    react_runner: Callable[..., Any] | None = None,
    checkpointer: Any | None = None,
):
    """Construct a real in-memory LangGraph CompiledStateGraph from a WorkflowSchema."""
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph, add_messages
        from langgraph.types import interrupt
    except ImportError as exc:
        raise RuntimeError(
            "langgraph is required for build_state_graph(). Install with: pip install langgraph-blocks[runtime]"
        ) from exc

    wf = workflow if isinstance(workflow, WorkflowSchema) else parse_workflow(workflow)
    tool_map = tools or {}

    state_annotations: dict[str, Any] = {"messages": Annotated[list[Any], add_messages]}
    for key in collect_state_vars(wf):
        state_annotations[key] = int if key.startswith("_loop_") else Any
    DynamicState = TypedDict("AgentState", state_annotations, total=False)  # type: ignore[misc]

    builder = StateGraph(DynamicState)
    counter = 0

    def next_node(prefix: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> str:
        nonlocal counter
        counter += 1
        name = f"{sanitize_identifier(prefix, 'node')}_{counter}"
        builder.add_node(name, fn)
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
                builder.add_edge(prev_exit, entry)
            prev_exit = exit_node
        assert first_entry is not None and prev_exit is not None
        return first_entry, prev_exit

    def lower_block(blk: BlockNode) -> tuple[str, str]:
        cfg = blk.config

        if blk.type == "ask_llm":
            step_name = str(cfg.get("name", "ask_llm"))
            out_var = sanitize_identifier(str(cfg.get("output_var", "llm_output")))
            model = str(cfg.get("model", "gpt-4.1-mini"))
            prompt_tpl = str(cfg.get("prompt", ""))

            def node_fn(s: dict[str, Any]) -> dict[str, Any]:
                prompt = render_template(prompt_tpl, s)
                return {out_var: _invoke_llm(llm, prompt, s, step_name, model)}

            n_name = next_node(step_name, node_fn)
            return n_name, n_name

        if blk.type == "agent_loop":
            step_name = str(cfg.get("name", "agent_loop"))
            out_var = sanitize_identifier(str(cfg.get("output_var", "agent_draft")))
            tools_list = [t.strip() for t in str(cfg.get("tools", "")).split(",") if t.strip()]
            max_steps = max(1, int(cfg.get("max_steps", 5)))
            prompt_tpl = str(cfg.get("prompt", ""))
            model = str(cfg.get("model", "claude-sonnet"))

            def node_fn(s: dict[str, Any]) -> dict[str, Any]:
                prompt = render_template(prompt_tpl, s)
                if react_runner is not None:
                    res = react_runner(prompt=prompt, tools=tools_list, max_steps=max_steps, state=s)
                else:
                    res = _invoke_llm(llm, prompt, s, step_name, model)
                return {out_var: res}

            n_name = next_node(step_name, node_fn)
            return n_name, n_name

        if blk.type == "call_tool":
            tool_name = str(cfg.get("tool_name", "tool"))
            arg_expr = str(cfg.get("arg_expr", ""))
            out_var = sanitize_identifier(str(cfg.get("output_var", "tool_result")))

            def node_fn(s: dict[str, Any]) -> dict[str, Any]:
                arg_val = render_template(arg_expr, s)
                return {out_var: _invoke_tool(tool_map, tool_name, arg_val, s)}

            n_name = next_node(tool_name, node_fn)
            return n_name, n_name

        if blk.type == "respond":
            out_var = sanitize_identifier(str(cfg.get("output_var", "final_response")))
            template = str(cfg.get("template", ""))
            n_name = next_node("respond", lambda s: {out_var: render_template(template, s)})
            return n_name, n_name

        if blk.type == "if_else":
            left_var = sanitize_identifier(str(cfg.get("left_var", "intent")))
            op = str(cfg.get("operator", "=="))
            right_val = cfg.get("right_val", "")
            gate = next_node("if_gate", lambda s: {})
            join = next_node("if_join", lambda s: {})
            then_res = lower_sequence(blk.branches.get("then", []))
            else_res = lower_sequence(blk.branches.get("else", []))
            builder.add_conditional_edges(
                gate,
                lambda s: "true" if eval_condition(s.get(left_var), op, right_val) else "false",
                {"true": then_res[0] if then_res else join, "false": else_res[0] if else_res else join},
            )
            if then_res:
                builder.add_edge(then_res[1], join)
            if else_res:
                builder.add_edge(else_res[1], join)
            return gate, join

        if blk.type == "human_approval":
            question = str(cfg.get("question", "Approve action?"))
            out_var = sanitize_identifier(str(cfg.get("output_var", "approval")))

            def human_fn(s: dict[str, Any]) -> dict[str, Any]:
                prompt = render_template(question, s)
                decision = interrupt({"question": prompt, "state_preview": dict(s)})
                approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
                return {out_var: approved}

            gate = next_node("human_gate", human_fn)
            join = next_node("hitl_join", lambda s: {})
            app_res = lower_sequence(blk.branches.get("approved", []))
            rej_res = lower_sequence(blk.branches.get("rejected", []))
            builder.add_conditional_edges(
                gate,
                lambda s: "approved" if s.get(out_var) else "rejected",
                {"approved": app_res[0] if app_res else join, "rejected": rej_res[0] if rej_res else join},
            )
            if app_res:
                builder.add_edge(app_res[1], join)
            if rej_res:
                builder.add_edge(rej_res[1], join)
            return gate, join

        if blk.type == "repeat_until":
            loop_key = sanitize_identifier(f"_loop_{blk.id}_count")
            cond_var = sanitize_identifier(str(cfg.get("condition_var", "quality_score")))
            op = str(cfg.get("operator", ">="))
            target_val = cfg.get("target_val", "8")
            max_iters = max(1, int(cfg.get("max_iterations", 3)))

            guard = next_node("loop_guard", lambda s: {loop_key: int(s.get(loop_key, 0)) + 1})
            done = next_node("loop_done", lambda s: {})
            body_res = lower_sequence(blk.branches.get("body", []))
            if body_res:
                builder.add_edge(body_res[1], guard)
                builder.add_conditional_edges(
                    guard,
                    lambda s: "done"
                    if (
                        (int(s.get(loop_key, 0)) > 1 and eval_condition(s.get(cond_var), op, target_val))
                        or int(s.get(loop_key, 0)) > max_iters
                    )
                    else "again",
                    {"again": body_res[0], "done": done},
                )
            else:
                builder.add_edge(guard, done)
            return guard, done

        if blk.type == "parallel":
            split = next_node("fan_out", lambda s: {})
            join = next_node("fan_in", lambda s: {})
            lane_a = lower_sequence(blk.branches.get("laneA", []))
            lane_b = lower_sequence(blk.branches.get("laneB", []))
            active = [lane for lane in (lane_a, lane_b) if lane is not None]
            if not active:
                builder.add_edge(split, join)
            elif len(active) == 1:
                builder.add_edge(split, active[0][0])
                builder.add_edge(active[0][1], join)
            else:
                for entry, _ in active:
                    builder.add_edge(split, entry)
                builder.add_edge([ex for _, ex in active], join)
            return split, join

        raise ValueError(f"Unsupported block type: {blk.type}")

    root_res = lower_sequence(wf.blocks)
    if root_res is None:
        noop = next_node("noop", lambda s: {})
        builder.add_edge(START, noop)
        builder.add_edge(noop, END)
    else:
        builder.add_edge(START, root_res[0])
        builder.add_edge(root_res[1], END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)


@dataclass
class TraceEvent:
    """Single executed block record from WorkflowRunner."""

    block_id: str
    block_type: str
    label: str
    detail: str
    status: str = "done"

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "label": self.label,
            "detail": self.detail,
            "status": self.status,
        }


@dataclass
class RunResult:
    """Result of starting or resuming a workflow via WorkflowRunner."""

    status: str  # "done" | "paused"
    state: dict[str, Any]
    trace: list[TraceEvent] = field(default_factory=list)
    interrupt_payload: dict[str, Any] | None = None
    resume_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "state": self.state,
            "trace": [t.to_dict() for t in self.trace],
            "interrupt_payload": self.interrupt_payload,
            "resume_token": self.resume_token,
        }


@dataclass
class _PausedExecution:
    workflow: WorkflowSchema
    state: dict[str, Any]
    trace: list[TraceEvent]
    frames: list[tuple[list[BlockNode], int]]
    approval_block: BlockNode
    llm: Any | None
    tools: dict[str, Callable[..., Any]]
    react_runner: Callable[..., Any] | None


class WorkflowRunner:
    """Dependency-free step-trace runner with HITL pause/resume for testing & Studio."""

    def __init__(self) -> None:
        self._paused: dict[str, _PausedExecution] = {}

    def start(
        self,
        workflow: WorkflowSchema | dict[str, Any],
        input_data: dict[str, Any] | str,
        *,
        llm: Any | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        react_runner: Callable[..., Any] | None = None,
    ) -> RunResult:
        wf = workflow if isinstance(workflow, WorkflowSchema) else parse_workflow(workflow)
        in_key = sanitize_identifier(wf.trigger.input_var or "user_query")
        if isinstance(input_data, str):
            state: dict[str, Any] = {in_key: input_data}
        else:
            state = dict(input_data)
            if in_key not in state and "query" in state:
                state[in_key] = state["query"]

        trace = [
            TraceEvent(
                block_id="trigger",
                block_type="trigger",
                label="Trigger",
                detail=f"Started {wf.trigger.name} with {in_key}={state.get(in_key)!r}",
            )
        ]
        frames: list[tuple[list[BlockNode], int]] = [(list(wf.blocks), 0)]
        return self._drive(wf, state, trace, frames, llm=llm, tools=tools or {}, react_runner=react_runner)

    def resume(self, resume_token: str, decision: bool | dict[str, Any]) -> RunResult:
        paused = self._paused.pop(resume_token, None)
        if paused is None:
            raise KeyError(f"Unknown or expired resume token: {resume_token}")

        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        blk = paused.approval_block
        out_var = sanitize_identifier(str(blk.config.get("output_var", "approval")))
        paused.state[out_var] = approved
        paused.trace.append(
            TraceEvent(
                block_id=blk.id,
                block_type="human_approval",
                label="Resumed Thread",
                detail=f"Reviewer chose {'Approve' if approved else 'Reject'} ({out_var}={approved})",
            )
        )
        branch = blk.branches.get("approved" if approved else "rejected", [])
        if branch:
            paused.frames.append((list(branch), 0))
        return self._drive(
            paused.workflow,
            paused.state,
            paused.trace,
            paused.frames,
            llm=paused.llm,
            tools=paused.tools,
            react_runner=paused.react_runner,
        )

    def _drive(
        self,
        wf: WorkflowSchema,
        state: dict[str, Any],
        trace: list[TraceEvent],
        frames: list[tuple[list[BlockNode], int]],
        *,
        llm: Any | None,
        tools: dict[str, Callable[..., Any]],
        react_runner: Callable[..., Any] | None,
    ) -> RunResult:
        while frames:
            seq, idx = frames[-1]
            if idx >= len(seq):
                frames.pop()
                continue

            blk = seq[idx]
            frames[-1] = (seq, idx + 1)
            cfg = blk.config

            if blk.type == "ask_llm":
                step_name = str(cfg.get("name", "ask_llm"))
                out_var = sanitize_identifier(str(cfg.get("output_var", "llm_output")))
                model = str(cfg.get("model", "gpt-4.1-mini"))
                prompt = render_template(str(cfg.get("prompt", "")), state)
                val = _invoke_llm(llm, prompt, state, step_name, model)
                state[out_var] = val
                trace.append(TraceEvent(blk.id, blk.type, f"Ask LLM ({step_name})", f"Set {out_var} = {val!r}"))

            elif blk.type == "agent_loop":
                step_name = str(cfg.get("name", "agent_loop"))
                out_var = sanitize_identifier(str(cfg.get("output_var", "agent_draft")))
                max_steps = max(1, int(cfg.get("max_steps", 5)))
                tools_list = [t.strip() for t in str(cfg.get("tools", "")).split(",") if t.strip()]
                prompt = render_template(str(cfg.get("prompt", "")), state)
                if react_runner is not None:
                    val = react_runner(prompt=prompt, tools=tools_list, max_steps=max_steps, state=state)
                else:
                    val = f"Tool-grounded answer using {tools_list} (max_steps={max_steps})"
                state[out_var] = val
                trace.append(TraceEvent(blk.id, blk.type, f"Agent Loop ({step_name})", f"Set {out_var}"))

            elif blk.type == "call_tool":
                tool_name = str(cfg.get("tool_name", "tool"))
                out_var = sanitize_identifier(str(cfg.get("output_var", "tool_result")))
                arg_val = render_template(str(cfg.get("arg_expr", "")), state)
                val = _invoke_tool(tools, tool_name, arg_val, state)
                state[out_var] = val
                trace.append(TraceEvent(blk.id, blk.type, f"Tool: {tool_name}", f"Saved output to {out_var}"))

            elif blk.type == "respond":
                out_var = sanitize_identifier(str(cfg.get("output_var", "final_response")))
                val = render_template(str(cfg.get("template", "")), state)
                state[out_var] = val
                trace.append(TraceEvent(blk.id, blk.type, "Final Reply", f"Set {out_var}"))

            elif blk.type == "if_else":
                left_var = sanitize_identifier(str(cfg.get("left_var", "intent")))
                op = str(cfg.get("operator", "=="))
                right_val = cfg.get("right_val", "")
                matched = eval_condition(state.get(left_var), op, right_val)
                branch_name = "then" if matched else "else"
                trace.append(
                    TraceEvent(
                        blk.id,
                        blk.type,
                        "If / Else",
                        f"{left_var}={state.get(left_var)!r} {op} {right_val!r} -> {branch_name}",
                    )
                )
                branch = blk.branches.get(branch_name, [])
                if branch:
                    frames.append((list(branch), 0))

            elif blk.type == "repeat_until":
                loop_key = sanitize_identifier(f"_loop_{blk.id}_count")
                cond_var = sanitize_identifier(str(cfg.get("condition_var", "quality_score")))
                op = str(cfg.get("operator", ">="))
                target_val = cfg.get("target_val", "8")
                max_iters = max(1, int(cfg.get("max_iterations", 3)))
                cur_iter = int(state.get(loop_key, 0))
                if cur_iter > 0 and eval_condition(state.get(cond_var), op, target_val):
                    trace.append(TraceEvent(blk.id, blk.type, "Loop Exit", f"{cond_var}={state.get(cond_var)!r} satisfied"))
                    continue
                if cur_iter >= max_iters:
                    trace.append(TraceEvent(blk.id, blk.type, "Loop Cap Reached", f"Stopped at max_iterations={max_iters}"))
                    continue
                state[loop_key] = cur_iter + 1
                trace.append(
                    TraceEvent(
                        blk.id,
                        blk.type,
                        f"Repeat Iteration #{cur_iter + 1}",
                        f"Max iterations: {max_iters}",
                    )
                )
                body = blk.branches.get("body", [])
                if body:
                    # Re-queue this repeat_until block after body finishes
                    frames[-1] = (seq, idx)
                    frames.append((list(body), 0))

            elif blk.type == "parallel":
                lane_a = blk.branches.get("laneA", [])
                lane_b = blk.branches.get("laneB", [])
                trace.append(TraceEvent(blk.id, blk.type, "Parallel Fan-Out", "Running Branch A & Branch B"))

                def run_lane(lane_nodes: list[BlockNode]) -> tuple[dict[str, Any], list[TraceEvent]]:
                    sub_wf = WorkflowSchema(trigger=wf.trigger, blocks=lane_nodes)
                    sub_state = dict(state)
                    sub_trace: list[TraceEvent] = []
                    self._drive(
                        sub_wf,
                        sub_state,
                        sub_trace,
                        [(list(lane_nodes), 0)],
                        llm=llm,
                        tools=tools,
                        react_runner=react_runner,
                    )
                    return sub_state, sub_trace

                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_a = pool.submit(run_lane, lane_a) if lane_a else None
                    fut_b = pool.submit(run_lane, lane_b) if lane_b else None
                    if fut_a is not None:
                        st_a, tr_a = fut_a.result()
                        for k, v in st_a.items():
                            if k not in state or state[k] != v:
                                state[k] = v
                        trace.extend(tr_a)
                    if fut_b is not None:
                        st_b, tr_b = fut_b.result()
                        for k, v in st_b.items():
                            if k not in state or state[k] != v:
                                state[k] = v
                        trace.extend(tr_b)

                trace.append(TraceEvent(blk.id, blk.type, "Parallel Fan-In", "Joined parallel branches into state"))

            elif blk.type == "human_approval":
                question = render_template(str(cfg.get("question", "Approve action?")), state)
                token = str(uuid.uuid4())
                trace.append(
                    TraceEvent(
                        blk.id,
                        blk.type,
                        "Human Approval",
                        f"Paused at interrupt(): {question}",
                        status="paused",
                    )
                )
                self._paused[token] = _PausedExecution(
                    workflow=wf,
                    state=state,
                    trace=trace,
                    frames=frames,
                    approval_block=blk,
                    llm=llm,
                    tools=tools,
                    react_runner=react_runner,
                )
                return RunResult(
                    status="paused",
                    state=dict(state),
                    trace=list(trace),
                    interrupt_payload={"question": question},
                    resume_token=token,
                )

        trace.append(TraceEvent("end", "end", "END", "Workflow completed & checkpoint saved."))
        return RunResult(status="done", state=dict(state), trace=list(trace))
