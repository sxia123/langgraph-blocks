"""Tests for schema parsing, normalization, and safety validation."""

import pytest

from langgraph_blocks.presets import get_builtin_presets
from langgraph_blocks.schema import (
    ALL_BLOCK_TYPES,
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


def test_expanded_18_block_catalog_validation_and_state_vars() -> None:
    assert len(ALL_BLOCK_TYPES) == 18
    workflow = {
        "trigger": {"name": "expanded_flow", "input_var": "user_query", "checkpointer": "MemorySaver"},
        "blocks": [
            {
                "id": "g1",
                "type": "guardrail",
                "config": {"target_var": "user_query", "rule_type": "no_pii", "output_var": "safe_ok"},
                "branches": {
                    "pass": [
                        {
                            "id": "ej1",
                            "type": "extract_json",
                            "config": {"source_var": "user_query", "schema_fields": "topic: str", "output_var": "parsed"},
                        },
                        {
                            "id": "rag1",
                            "type": "rag_retrieve",
                            "config": {"collection": "docs", "query_expr": "{{user_query}}", "top_k": 2, "output_var": "kb_hits"},
                        },
                        {
                            "id": "r1",
                            "type": "llm_router",
                            "config": {"label_a": "billing", "label_b": "tech", "output_var": "route_choice"},
                            "branches": {
                                "route_a": [
                                    {"id": "sub1", "type": "sub_workflow", "config": {"workflow_name": "billing", "output_var": "sub_out"}}
                                ],
                                "route_b": [
                                    {"id": "http1", "type": "http_request", "config": {"method": "POST", "url_expr": "https://api.test/t", "output_var": "http_res"}}
                                ],
                                "fallback": [
                                    {"id": "st1", "type": "set_state", "config": {"target_var": "triage_status", "value_expr": "manual"}}
                                ],
                            },
                        },
                        {
                            "id": "fe1",
                            "type": "for_each",
                            "config": {"list_var": "kb_hits", "item_var": "doc_item", "max_items": 5},
                            "branches": {
                                "body": [
                                    {"id": "mem1", "type": "append_memory", "config": {"role": "assistant", "content_expr": "{{doc_item}}", "target_var": "messages"}}
                                ]
                            },
                        },
                        {
                            "id": "tf1",
                            "type": "try_fallback",
                            "config": {"error_var": "err_msg"},
                            "branches": {"try": [], "fallback": []},
                        },
                    ],
                    "blocked": [],
                },
            }
        ],
    }
    wf = parse_workflow(workflow)
    svars = collect_state_vars(wf)
    assert {"safe_ok", "parsed", "kb_hits", "route_choice", "sub_out", "http_res", "triage_status", "doc_item", "_for_fe1_idx", "err_msg"} <= set(svars)
    assert "messages" not in svars

    bad_expanded = {
        "trigger": {"name": "bad_exp", "input_var": "q", "checkpointer": "MemorySaver"},
        "blocks": [
            {"id": "b_guard", "type": "guardrail", "config": {"rule_type": "invalid_rule"}, "branches": {"pass": []}},
            {"id": "b_for", "type": "for_each", "config": {"max_items": 0}, "branches": {"body": []}},
            {"id": "b_rag", "type": "rag_retrieve", "config": {"top_k": 0}},
        ],
    }
    with pytest.raises(ValidationError) as exc_info:
        parse_workflow(bad_expanded)
    err_text = " | ".join(exc_info.value.errors)
    assert "unsupported guardrail rule_type" in err_text
    assert "missing branch 'blocked'" in err_text
    assert "max_items >= 1" in err_text
    assert "top_k >= 1" in err_text

    bad_ai = {
        "trigger": {"name": "bad_ai", "input_var": "q", "checkpointer": "MemorySaver"},
        "blocks": [
            {"id": "a1", "type": "ask_llm", "config": {"model": "claude-sonnet-4-5", "temperature": 2.5, "maxTokens": 0}},
        ],
    }
    with pytest.raises(ValidationError) as ai_exc:
        parse_workflow(bad_ai)
    ai_err = " | ".join(ai_exc.value.errors)
    assert "temperature must be between 0.0 and 2.0" in ai_err
    assert "max_tokens must be >= 1" in ai_err
