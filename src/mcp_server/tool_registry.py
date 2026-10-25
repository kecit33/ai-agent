"""工具注册表：一处定义，三处复用（MCP tools/list、OpenAI function schema、实际执行）。

x-requires-confirm 是本项目在标准 MCP inputSchema 之外附加的扩展字段：
标记该工具属于"写操作/危险操作"，客户端在执行前必须走 human-in-the-loop 二次确认。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..common.errors import UserInputError

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    requires_confirm: bool = False   # True => 执行前必须人工确认
    category: str = "general"

    def to_mcp_tool(self) -> dict[str, Any]:
        """转成 MCP tools/list 规定的工具描述格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                **self.input_schema,
                "x-requires-confirm": self.requires_confirm,
            },
        }

    def to_openai_function(self) -> dict[str, Any]:
        """转成 OpenAI Chat Completions 的 function/tool 描述格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> None:
    """轻量校验：只查 required 与基础类型。

    不做完整 JSON Schema 校验（会引入 jsonschema 依赖），这里覆盖 95% 的模型参数错误场景，
    更深的语义校验交给工具 handler 自己抛 UserInputError。
    """
    if not isinstance(args, dict):
        raise UserInputError("工具参数必须是 JSON 对象")

    required = schema.get("required", []) or []
    for key in required:
        if key not in args or args[key] is None:
            raise UserInputError(f"缺少必填参数: {key}")

    properties = schema.get("properties", {}) or {}
    for key, value in args.items():
        if key not in properties:
            raise UserInputError(f"出现未定义参数: {key}")
        expected = properties[key].get("type")
        ok = True
        if expected == "string":
            ok = isinstance(value, str)
        elif expected == "integer":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected == "number":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected == "boolean":
            ok = isinstance(value, bool)
        elif expected == "array":
            ok = isinstance(value, list)
        elif expected == "object":
            ok = isinstance(value, dict)
        if not ok:
            raise UserInputError(f"参数 {key} 类型应为 {expected}，实际为 {type(value).__name__}")


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重名: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(name)
        return spec

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def list_mcp_tools(self) -> list[dict[str, Any]]:
        return [self._tools[n].to_mcp_tool() for n in self._tools]

    def list_openai_functions(self) -> list[dict[str, Any]]:
        return [self._tools[n].to_openai_function() for n in self._tools]

    async def call(self, name: str, args: dict[str, Any], timeout_sec: float = 20.0) -> dict[str, Any]:
        """执行工具，并强制施加超时上限——没有超时的工具调用是生产事故的第一来源。"""
        spec = self.get(name)
        validate_args(spec.input_schema, args)
        try:
            return await asyncio.wait_for(spec.handler(args), timeout=timeout_sec)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"工具 {name} 执行超时（>{timeout_sec}s）") from exc
