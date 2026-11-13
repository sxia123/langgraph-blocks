"""Canonical JSON AST schema, normalization, and validation for LangGraph Blocks."""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any

ALLOWED_CHECKPOINTERS = frozenset({"MemorySaver", "PostgresSaver", "SqliteSaver"})
ALLOWED_OPERATORS = frozenset({"==", "!=", ">=", ">", "<=", "<", "in"})

CONTAINER_BRANCHES: dict[str, tuple[str, ...]] = {
    "if_else": ("then", "else"),
    "repeat_until": ("body",),
    "parallel": ("laneA", "laneB"),
    "human_approval": ("approved", "rejected"),
}

LEAF_TYPES = frozenset({"ask_llm", "agent_loop", "call_tool", "respond"})
ALL_BLOCK_TYPES = frozenset(CONTAINER_BRANCHES.keys()) | LEAF_TYPES

_CAMEL_TO_SNAKE = {
    "inputVar": "input_var",
    "outputVar": "output_var",
    "outputFormat": "output_format",
    "contextMode": "context_mode",
    "maxSteps": "max_steps",
    "maxIterations": "max_iterations",
    "leftVar": "left_var",
    "rightVal": "right_val",
    "conditionVar": "condition_var",
    "targetVal": "target_val",
    "toolName": "tool_name",
    "argExpr": "arg_expr",
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

            expected_branches = CONTAINER_BRANCHES.get(node.type)
            if expected_branches is not None:
                for b_key in expected_branches:
                    if b_key not in node.branches:
                        errors.append(f"{loc}: missing branch '{b_key}'.")
                    else:
                        walk(node.branches[b_key], f"{loc}.{b_key}")
            elif node.branches:
                errors.append(f"{loc}: leaf block '{node.type}' cannot contain branches.")

    walk(workflow.blocks, "blocks")
    return errors


def collect_state_vars(workflow: WorkflowSchema) -> list[str]:
    """Collect ordered, sanitized state keys needed by the compiled graph."""
    keys: list[str] = []
    seen: set[str] = set()

    def add_key(raw: str) -> None:
        key = sanitize_identifier(raw)
        if key not in seen:
            seen.add(key)
            keys.append(key)

    add_key(workflow.trigger.input_var or "user_query")

    def walk(nodes: list[BlockNode]) -> None:
        for node in nodes:
            out_var = node.config.get("output_var")
            if out_var:
                add_key(str(out_var))
            if node.type == "repeat_until":
                add_key(f"_loop_{sanitize_identifier(node.id, 'loop')}_count")
                cond_var = node.config.get("condition_var")
                if cond_var:
                    add_key(str(cond_var))
            elif node.type == "if_else":
                left_var = node.config.get("left_var")
                if left_var:
                    add_key(str(left_var))
            for children in node.branches.values():
                walk(children)

    walk(workflow.blocks)
    return keys
