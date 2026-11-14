"""Canonical JSON AST schema, normalization, and validation for LangGraph Blocks."""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any

ALLOWED_CHECKPOINTERS = frozenset({"MemorySaver", "PostgresSaver", "SqliteSaver"})
ALLOWED_OPERATORS = frozenset({"==", "!=", ">=", ">", "<=", "<", "in"})
ALLOWED_GUARDRAIL_RULES = frozenset({"not_empty", "no_pii", "max_length", "regex"})
ALLOWED_LLM_MODES = frozenset({"api", "custom_endpoint"})
ALLOWED_API_PROVIDERS = frozenset({"openai", "anthropic", "gemini"})
ALLOWED_FORK_MODES = frozenset({"parallel", "conditional"})

CONTAINER_BRANCHES: dict[str, tuple[str, ...]] = {
    "if_else": ("then", "else"),
    "repeat_until": ("body",),
    "for_each": ("body",),
    "parallel": ("laneA", "laneB"),
    "try_fallback": ("try", "fallback"),
    "human_approval": ("approved", "rejected"),
    "llm_router": ("route_a", "route_b", "fallback"),
    "guardrail": ("pass", "blocked"),
}

LEAF_TYPES = frozenset(
    {
        "ask_llm",
        "extract_json",
        "agent_loop",
        "rag_retrieve",
        "set_state",
        "append_memory",
        "call_tool",
        "http_request",
        "sub_workflow",
        "respond",
    }
)
ALL_BLOCK_TYPES = frozenset(CONTAINER_BRANCHES.keys()) | LEAF_TYPES

_CAMEL_TO_SNAKE = {
    "inputVar": "input_var",
    "outputVar": "output_var",
    "outputFormat": "output_format",
    "contextMode": "context_mode",
    "maxSteps": "max_steps",
    "maxIterations": "max_iterations",
    "maxItems": "max_items",
    "leftVar": "left_var",
    "rightVal": "right_val",
    "conditionVar": "condition_var",
    "targetVal": "target_val",
    "targetVar": "target_var",
    "valueExpr": "value_expr",
    "contentExpr": "content_expr",
    "sourceVar": "source_var",
    "schemaFields": "schema_fields",
    "queryExpr": "query_expr",
    "topK": "top_k",
    "listVar": "list_var",
    "itemVar": "item_var",
    "errorVar": "error_var",
    "ruleType": "rule_type",
    "ruleArg": "rule_arg",
    "urlExpr": "url_expr",
    "bodyExpr": "body_expr",
    "workflowName": "workflow_name",
    "inputExpr": "input_expr",
    "toolName": "tool_name",
    "argExpr": "arg_expr",
    "systemPrompt": "system_prompt",
    "maxTokens": "max_tokens",
    "llmMode": "llm_mode",
    "apiProvider": "api_provider",
    "apiKeyEnv": "api_key_env",
    "endpointUrl": "endpoint_url",
    "forkMode": "fork_mode",
    "forkBranches": "fork_branches",
}


class ValidationError(ValueError):
    """Raised when a workflow AST fails structural or safety validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def sanitize_identifier(raw: str, default: str = "step") -> str:
    """Convert an arbitrary label into a valid, safe Python identifier."""
    raw_str = str(raw or default)
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw_str).lower()
    cleaned = re.sub(r"_+", "_", cleaned).rstrip("_")
    if not raw_str.startswith("_"):
        cleaned = cleaned.lstrip("_")
    if not cleaned.strip("_"):
        cleaned = default
    if cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_val"
    return cleaned


def _normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        target_key = _CAMEL_TO_SNAKE.get(key, key)
        normalized[target_key] = value
    return normalized


@dataclass
class TriggerConfig:
    """Entrypoint configuration for a compiled block workflow."""

    name: str = "agent_workflow"
    input_var: str = "user_query"
    checkpointer: str = "MemorySaver"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TriggerConfig:
        norm = _normalize_config(data)
        return cls(
            name=str(norm.get("name", "agent_workflow")),
            input_var=str(norm.get("input_var", "user_query")),
            checkpointer=str(norm.get("checkpointer", "MemorySaver")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_var": self.input_var,
            "checkpointer": self.checkpointer,
        }


@dataclass
class BlockNode:
    """Single node in the structured block AST."""

    id: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)
    branches: dict[str, list[BlockNode]] = field(default_factory=dict)

    @property
    def is_container(self) -> bool:
        return self.type in CONTAINER_BRANCHES

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockNode:
        if not isinstance(data, dict):
            raise ValidationError(["Each block must be a JSON object."])
        block_type = str(data.get("type", ""))
        raw_branches = data.get("branches") or {}
        branches: dict[str, list[BlockNode]] = {}
        if isinstance(raw_branches, dict):
            for key, children in raw_branches.items():
                branches[str(key)] = [cls.from_dict(child) for child in (children or [])]
        return cls(
            id=str(data.get("id", "")),
            type=block_type,
            config=_normalize_config(data.get("config")),
            branches=branches,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "config": dict(self.config),
        }
        if self.branches:
            payload["branches"] = {
                k: [child.to_dict() for child in children]
                for k, children in self.branches.items()
            }
        return payload


@dataclass
class WorkflowSchema:
    """Top-level workflow AST holding trigger metadata and block tree."""

    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    blocks: list[BlockNode] = field(default_factory=list)
    version: str = "0.1"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowSchema:
        if not isinstance(data, dict):
            raise ValidationError(["Workflow payload must be a JSON object."])
        trigger = TriggerConfig.from_dict(data.get("trigger"))
        raw_blocks = data.get("blocks") or []
        blocks = [BlockNode.from_dict(item) for item in raw_blocks]
        return cls(
            trigger=trigger,
            blocks=blocks,
            version=str(data.get("version", "0.1")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "trigger": self.trigger.to_dict(),
            "blocks": [b.to_dict() for b in self.blocks],
        }


def parse_workflow(data: dict[str, Any], *, validate: bool = True) -> WorkflowSchema:
    """Parse and optionally validate a raw workflow dictionary."""
    workflow = WorkflowSchema.from_dict(data)
    if validate:
        errors = validate_workflow(workflow)
        if errors:
            raise ValidationError(errors)
    return workflow


def validate_workflow(workflow: WorkflowSchema) -> list[str]:
    """Validate structural integrity and safety bounds; returns error messages."""
    errors: list[str] = []
    if not workflow.trigger.name.strip():
        errors.append("Trigger name must not be empty.")
    if not workflow.trigger.input_var.strip():
        errors.append("Trigger input_var must not be empty.")
    if workflow.trigger.checkpointer not in ALLOWED_CHECKPOINTERS:
        allowed = ", ".join(sorted(ALLOWED_CHECKPOINTERS))
        errors.append(
            f"Unsupported checkpointer '{workflow.trigger.checkpointer}'. Allowed: {allowed}."
        )

    seen_ids: set[str] = set()

    def walk(nodes: list[BlockNode], path: str) -> None:
        for idx, node in enumerate(nodes):
            loc = f"{path}[{idx}]({node.id or '?'})"
            if not node.id.strip():
                errors.append(f"{loc}: block id must not be empty.")
            elif node.id in seen_ids:
                errors.append(f"{loc}: duplicate block id '{node.id}'.")
            else:
                seen_ids.add(node.id)

            if node.type not in ALL_BLOCK_TYPES:
                errors.append(f"{loc}: unknown block type '{node.type}'.")
                continue

            if node.type == "agent_loop":
                try:
                    max_steps = int(node.config.get("max_steps", 0))
                except (TypeError, ValueError):
                    max_steps = 0
                if max_steps < 1:
                    errors.append(f"{loc}: agent_loop requires max_steps >= 1.")

            if node.type == "if_else":
                op = str(node.config.get("operator", "=="))
                if op not in ALLOWED_OPERATORS:
                    errors.append(f"{loc}: unsupported operator '{op}'.")

            if node.type == "repeat_until":
                op = str(node.config.get("operator", ">="))
                if op not in ALLOWED_OPERATORS:
                    errors.append(f"{loc}: unsupported operator '{op}'.")
                try:
                    max_iters = int(node.config.get("max_iterations", 0))
                except (TypeError, ValueError):
                    max_iters = 0
                if max_iters < 1:
                    errors.append(f"{loc}: repeat_until requires max_iterations >= 1.")

            if node.type == "for_each":
                try:
                    max_items = int(node.config.get("max_items", 10))
                except (TypeError, ValueError):
                    max_items = 0
                if max_items < 1:
                    errors.append(f"{loc}: for_each requires max_items >= 1.")

            if node.type == "rag_retrieve":
                try:
                    top_k = int(node.config.get("top_k", 3))
                except (TypeError, ValueError):
                    top_k = 0
                if top_k < 1:
                    errors.append(f"{loc}: rag_retrieve requires top_k >= 1.")

            if node.type == "guardrail":
                rule_type = str(node.config.get("rule_type", "not_empty"))
                if rule_type not in ALLOWED_GUARDRAIL_RULES:
                    errors.append(f"{loc}: unsupported guardrail rule_type '{rule_type}'.")

            if node.type in {"ask_llm", "extract_json", "agent_loop", "llm_router"}:
                llm_mode = str(node.config.get("llm_mode", "api") or "api")
                if llm_mode not in ALLOWED_LLM_MODES:
                    errors.append(
                        f"{loc}: llm_mode must be 'api' or 'custom_endpoint' (got '{llm_mode}')."
                    )
                api_provider = str(node.config.get("api_provider", "") or "")
                if api_provider and api_provider not in ALLOWED_API_PROVIDERS:
                    errors.append(
                        f"{loc}: unsupported api_provider '{api_provider}'."
                    )
                endpoint_url = str(node.config.get("endpoint_url", "") or "").strip()
                if endpoint_url and not (
                    endpoint_url.startswith("http://") or endpoint_url.startswith("https://")
                ):
                    errors.append(
                        f"{loc}: endpoint_url must start with 'http://' or 'https://'."
                    )
                if "temperature" in node.config and node.config["temperature"] not in (None, ""):
                    try:
                        temp_val = float(node.config["temperature"])
                    except (TypeError, ValueError):
                        temp_val = -1.0
                    if not (0.0 <= temp_val <= 2.0):
                        errors.append(f"{loc}: temperature must be between 0.0 and 2.0.")
                if "max_tokens" in node.config and node.config["max_tokens"] not in (None, ""):
                    try:
                        tok_val = int(node.config["max_tokens"])
                    except (TypeError, ValueError):
                        tok_val = 0
                    if tok_val < 1:
                        errors.append(f"{loc}: max_tokens must be >= 1.")

            if "fork_mode" in node.config and node.config["fork_mode"] not in (None, ""):
                f_mode = str(node.config["fork_mode"])
                if f_mode not in ALLOWED_FORK_MODES:
                    errors.append(
                        f"{loc}: fork_mode must be 'parallel' or 'conditional' (got '{f_mode}')."
                    )

            raw_fork = node.config.get("fork_branches")
            expected_branches = CONTAINER_BRANCHES.get(node.type)
            reserved_keys = set(expected_branches or ())
            declared_fork_keys: set[str] = set()
            if raw_fork is not None:
                if not isinstance(raw_fork, list):
                    errors.append(f"{loc}: fork_branches must be a list.")
                else:
                    for b_idx, item in enumerate(raw_fork):
                        if not isinstance(item, dict) or not str(item.get("key", "")).strip():
                            errors.append(f"{loc}: fork_branches[{b_idx}] must include a non-empty 'key'.")
                            continue
                        b_key = str(item["key"]).strip()
                        if b_key in reserved_keys:
                            errors.append(f"{loc}: fork branch key '{b_key}' collides with built-in branch.")
                        elif b_key in declared_fork_keys:
                            errors.append(f"{loc}: duplicate fork branch key '{b_key}'.")
                        else:
                            declared_fork_keys.add(b_key)
                        b_op = str(item.get("op", "==") or "==")
                        if not bool(item.get("is_default", False)) and b_op not in ALLOWED_OPERATORS:
                            errors.append(f"{loc}: unsupported fork condition operator '{b_op}'.")

            if expected_branches is not None:
                for b_key in expected_branches:
                    if b_key not in node.branches:
                        errors.append(f"{loc}: missing branch '{b_key}'.")
                    else:
                        walk(node.branches[b_key], f"{loc}.{b_key}")
                for extra_key, extra_nodes in node.branches.items():
                    if extra_key not in reserved_keys:
                        walk(extra_nodes, f"{loc}.{extra_key}")
            elif node.branches:
                for b_key, children in node.branches.items():
                    walk(children, f"{loc}.{b_key}")

    walk(workflow.blocks, "blocks")
    return errors


def collect_state_vars(workflow: WorkflowSchema) -> list[str]:
    """Collect ordered, sanitized state keys needed by the compiled graph."""
    keys: list[str] = []
    seen: set[str] = {"messages"}

    def add_key(raw: str) -> None:
        key = sanitize_identifier(raw)
        if key not in seen:
            seen.add(key)
            keys.append(key)

    add_key(workflow.trigger.input_var or "user_query")

    def walk(nodes: list[BlockNode]) -> None:
        for node in nodes:
            for field_name in ("output_var", "target_var", "error_var", "list_var", "item_var", "source_var"):
                val = node.config.get(field_name)
                if val:
                    add_key(str(val))
            if node.type == "repeat_until":
                add_key(f"_loop_{sanitize_identifier(node.id, 'loop')}_count")
                cond_var = node.config.get("condition_var")
                if cond_var:
                    add_key(str(cond_var))
            elif node.type == "for_each":
                add_key(f"_for_{sanitize_identifier(node.id, 'for')}_idx")
            elif node.type == "if_else":
                left_var = node.config.get("left_var")
                if left_var:
                    add_key(str(left_var))
            for f_spec in get_node_fork_branches(node):
                if not f_spec["is_default"] and f_spec["var"]:
                    add_key(f_spec["var"])
            for children in node.branches.values():
                walk(children)

    walk(workflow.blocks)
    return keys


def get_parallel_lanes(node: BlockNode) -> list[str]:
    """Return ordered branch keys belonging to a parallel container block."""
    if node.type != "parallel":
        return []
    fork_keys = {
        str(item["key"])
        for item in (node.config.get("fork_branches") or [])
        if isinstance(item, dict) and item.get("key")
    }
    lanes = ["laneA", "laneB"]
    for key in node.branches:
        if key not in lanes and key not in fork_keys:
            lanes.append(key)
    return lanes


def get_fork_mode(node: BlockNode) -> str:
    """Return the normalized execution mode ('parallel' or 'conditional') for a node's fork."""
    raw = str(node.config.get("fork_mode") or "parallel").strip()
    return "conditional" if raw == "conditional" else "parallel"


def get_node_fork_branches(node: BlockNode) -> list[dict[str, Any]]:
    """Return ordered normalized outgoing fork descriptors for any block."""
    reserved = set(CONTAINER_BRANCHES.get(node.type, ()))
    if node.type == "parallel":
        reserved.update(k for k in node.branches if k.startswith("lane"))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    default_var = str(
        node.config.get("output_var")
        or node.config.get("target_var")
        or node.config.get("error_var")
        or "user_query"
    )
    raw_list = node.config.get("fork_branches")
    if isinstance(raw_list, list):
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key or key in reserved or key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "key": key,
                    "label": str(item.get("label") or key),
                    "var": str(item.get("var") or default_var),
                    "op": str(item.get("op") or "=="),
                    "val": item.get("val", ""),
                    "is_default": bool(item.get("is_default", False)),
                }
            )
    for key in node.branches:
        if key in reserved or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "key": key,
                "label": key,
                "var": default_var,
                "op": "==",
                "val": key,
                "is_default": False,
            }
        )
    return out
