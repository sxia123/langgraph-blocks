"""Tests for in-memory build_state_graph and WorkflowRunner."""

import uuid

from langgraph.types import Command

from langgraph_blocks import WorkflowRunner, build_state_graph
from langgraph_blocks.presets import get_builtin_presets


def test_in_memory_build_state_graph_runs_on_real_langgraph() -> None:
    presets = get_builtin_presets()
    graph = build_state_graph(presets["support_refund_hitl"])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # General branch (no interrupt)
    general_res = graph.invoke({"user_query": "What is your return window?"}, config=config)
    assert "__interrupt__" not in general_res
    assert general_res["intent"] == "general"
    assert general_res["final_response"]

    # Refund branch (hits real interrupt -> resume)
    refund_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    paused = graph.invoke({"user_query": "Please refund order #8492"}, config=refund_cfg)
    assert "__interrupt__" in paused
    done = graph.invoke(Command(resume={"approved": True}), config=refund_cfg)
    assert "RF-991" in done["final_response"]


def test_workflow_runner_pause_resume_and_loop() -> None:
    presets = get_builtin_presets()
    runner = WorkflowRunner()

    res = runner.start(presets["support_refund_hitl"], "Refund order #8492")
    assert res.status == "paused"
    assert res.resume_token is not None
    assert "Approve refund" in (res.interrupt_payload or {}).get("question", "")

    final_res = runner.resume(res.resume_token, True)
    assert final_res.status == "done"
    assert "RF-991" in final_res.state["final_response"]

    loop_res = runner.start(presets["research_critique_loop"], "LangGraph")
    assert loop_res.status == "done"
    assert loop_res.state["quality_score"] >= 8
    assert loop_res.state["final_response"]
