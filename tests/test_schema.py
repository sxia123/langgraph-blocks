"""Tests for schema parsing, normalization, and safety validation."""

import pytest

from langgraph_blocks.presets import get_builtin_presets
from langgraph_blocks.schema import (
    ValidationError,
    collect_state_vars,
    parse_workflow,
    sanitize_identifier,
)


def test_parse_builtin_presets() -> None:
    presets = get_builtin_presets()
    for name, raw in presets.items():
        wf = parse_workflow(raw)
        assert wf.trigger.name, f"Preset {name} missing trigger name"
        assert wf.blocks, f"Preset {name} has no blocks"


def test_camel_case_normalization() -> None:
    raw = {
        "trigger": {"name": "demoFlow", "inputVar": "questionText", "checkpointer": "MemorySaver"},
        "blocks": [
            {
                "id": "llm_1",
                "type": "ask_llm",
                "config": {
                    "name": "stepOne",
                    "outputVar": "answerText",
                    "outputFormat": "text",
                    "prompt": "Answer {{questionText}}",
                },
            }
        ],
    }
    wf = parse_workflow(raw)
    assert wf.trigger.input_var == "questionText"
    assert wf.blocks[0].config["output_var"] == "answerText"
    assert wf.blocks[0].config["output_format"] == "text"


def test_validation_rejects_duplicate_ids_and_unsafe_bounds() -> None:
    bad = {
        "trigger": {"name": "bad", "input_var": "q", "checkpointer": "UnknownSaver"},
        "blocks": [
            {
                "id": "dup",
                "type": "repeat_until",
                "config": {"condition_var": "score", "operator": ">=", "target_val": "8", "max_iterations": 0},
                "branches": {"body": []},
            },
            {
                "id": "dup",
                "type": "agent_loop",
                "config": {"max_steps": 0},
            },
        ],
    }
    with pytest.raises(ValidationError) as exc_info:
        parse_workflow(bad)
    msg = " | ".join(exc_info.value.errors)
    assert "UnknownSaver" in msg
    assert "duplicate block id" in msg
    assert "max_iterations >= 1" in msg
    assert "max_steps >= 1" in msg


def test_collect_state_vars_and_sanitize_identifier() -> None:
    assert sanitize_identifier("return") == "return_val"
    assert sanitize_identifier("123 step") == "v_123_step"
    presets = get_builtin_presets()
    wf = parse_workflow(presets["research_critique_loop"])
    state_vars = collect_state_vars(wf)
    assert "user_query" in state_vars
    assert "web_notes" in state_vars
    assert "internal_notes" in state_vars
    assert "_loop_repeat_until_1_count" in state_vars
