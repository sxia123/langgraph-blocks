"""Built-in starter workflow presets for LangGraph Blocks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "support_refund_hitl": {
        "version": "0.1",
        "trigger": {
            "name": "support_refund_flow",
            "input_var": "user_query",
            "checkpointer": "MemorySaver",
        },
        "blocks": [
            {
                "id": "ask_llm_1",
                "type": "ask_llm",
                "config": {
                    "name": "classify_intent",
                    "model": "gpt-4.1-mini",
                    "output_format": "structured",
                    "context_mode": "thread_messages",
                    "output_var": "intent",
                    "prompt": "Classify {{user_query}} as 'refund' or 'general'.",
                },
            },
            {
                "id": "if_else_1",
                "type": "if_else",
                "config": {
                    "left_var": "intent",
                    "operator": "==",
                    "right_val": "refund",
                },
                "branches": {
                    "then": [
                        {
                            "id": "call_tool_1",
                            "type": "call_tool",
                            "config": {
                                "tool_name": "lookup_order",
                                "arg_expr": "{{user_query}}",
                                "output_var": "order_record",
                            },
                        },
                        {
                            "id": "human_approval_1",
                            "type": "human_approval",
                            "config": {
                                "question": "Approve refund based on {{order_record}}?",
                                "output_var": "approval",
                            },
                            "branches": {
                                "approved": [
                                    {
                                        "id": "call_tool_2",
                                        "type": "call_tool",
                                        "config": {
                                            "tool_name": "issue_refund",
                                            "arg_expr": "{{order_record}}",
                                            "output_var": "refund_receipt",
                                        },
                                    },
                                    {
                                        "id": "respond_1",
                                        "type": "respond",
                                        "config": {
                                            "template": "Refund processed: {{refund_receipt}}",
                                            "output_var": "final_response",
                                        },
                                    },
                                ],
                                "rejected": [
                                    {
                                        "id": "respond_2",
                                        "type": "respond",
                                        "config": {
                                            "template": "Sorry, your refund request was declined after review.",
                                            "output_var": "final_response",
                                        },
                                    }
                                ],
                            },
                        },
                    ],
                    "else": [
                        {
                            "id": "agent_loop_1",
                            "type": "agent_loop",
                            "config": {
                                "name": "support_answer_agent",
                                "model": "claude-sonnet",
                                "tools": "search_kb",
                                "max_steps": 4,
                                "output_var": "answer_draft",
                                "prompt": "Answer {{user_query}} accurately using KB search.",
                            },
                        },
                        {
                            "id": "respond_3",
                            "type": "respond",
                            "config": {
                                "template": "{{answer_draft}}",
                                "output_var": "final_response",
                            },
                        },
                    ],
                },
            },
        ],
    },
    "research_critique_loop": {
        "version": "0.1",
        "trigger": {
            "name": "research_critique_flow",
            "input_var": "user_query",
            "checkpointer": "MemorySaver",
        },
        "blocks": [
            {
                "id": "parallel_1",
                "type": "parallel",
                "config": {"label": "Gather Context"},
                "branches": {
                    "laneA": [
                        {
                            "id": "call_tool_web",
                            "type": "call_tool",
                            "config": {
                                "tool_name": "web_search",
                                "arg_expr": "{{user_query}}",
                                "output_var": "web_notes",
                            },
                        }
                    ],
                    "laneB": [
                        {
                            "id": "call_tool_kb",
                            "type": "call_tool",
                            "config": {
                                "tool_name": "search_kb",
                                "arg_expr": "{{user_query}}",
                                "output_var": "internal_notes",
                            },
                        }
                    ],
                },
            },
            {
                "id": "repeat_until_1",
                "type": "repeat_until",
                "config": {
                    "condition_var": "quality_score",
                    "operator": ">=",
                    "target_val": "8",
                    "max_iterations": 3,
                },
                "branches": {
                    "body": [
                        {
                            "id": "ask_llm_write",
                            "type": "ask_llm",
                            "config": {
                                "name": "write_brief",
                                "model": "claude-sonnet",
                                "output_format": "text",
                                "output_var": "draft_brief",
                                "prompt": "Synthesize {{web_notes}} and {{internal_notes}} into a concise brief.",
                            },
                        },
                        {
                            "id": "ask_llm_score",
                            "type": "ask_llm",
                            "config": {
                                "name": "score_brief",
                                "model": "gpt-4.1-mini",
                                "output_format": "structured",
                                "output_var": "quality_score",
                                "prompt": "Score {{draft_brief}} from 1-10 for accuracy and completeness.",
                            },
                        },
                    ]
                },
            },
            {
                "id": "respond_final",
                "type": "respond",
                "config": {
                    "template": "{{draft_brief}}",
                    "output_var": "final_response",
                },
            },
        ],
    },
}


def get_builtin_presets() -> dict[str, dict[str, Any]]:
    """Return deep copies of the built-in workflow presets."""
    return deepcopy(BUILTIN_PRESETS)
