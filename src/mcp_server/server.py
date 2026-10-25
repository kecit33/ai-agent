"""MCP Server 主循环（stdio 传输 + JSON-RPC 2.0）。

启动方式（由 Agent 客户端拉起，也可单独手工跑）：
    python -m src.mcp_server.server

三条铁律：
1. stdout 只允许写协议消息，日志一律走 stderr（否则客户端解析会炸）；
2. 任何异常都要变成 JSON-RPC error 或 isError 结果，绝不能让进程崩溃退出；
3. 每个工具执行都有超时上限。
"""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from typing import Any

from ..common.config import load_config
from ..common.errors import AgentError
from ..common.logging_setup import get_logger, setup_logging
from . import protocol as P
from .tool_registry import ToolRegistry, ToolSpec
from .tools_code import code_run_python
from .tools_fs import fs_analyze_file, fs_list_dir, fs_read_file, fs_write_file
from .tools_net import http_get_json, time_now, weather_get
from .tools_todo import init_db, todo_add, todo_done, todo_list, todo_remove

SERVER_NAME = "homework-mcp-server"
SERVER_VERSION = "1.0.0"

_LOG = get_logger("mcp_server")


def build_registry() -> ToolRegistry:
    """集中声明全部工具。新增能力只需在这里加一条 ToolSpec。"""
    reg = ToolRegistry()

    reg.register(ToolSpec(
        name="fs_list_dir",
        description="列出工作区目录内容（受沙箱限制，只能访问 workspace 目录）",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 workspace 的路径，默认 '.'"},
                "recursive": {"type": "boolean", "description": "是否递归子目录，默认 false"},
            },
            "required": [],
        },
        handler=fs_list_dir,
        category="file",
    ))
    reg.register(ToolSpec(
        name="fs_read_file",
        description="读取工作区内的文本文件内容（自动截断，避免撑爆上下文）",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，相对 workspace 或绝对路径"},
                "max_chars": {"type": "integer", "description": "最多读取字符数，默认 8000"},
            },
            "required": ["path"],
        },
        handler=fs_read_file,
        category="file",
    ))
    reg.register(ToolSpec(
        name="fs_write_file",
        description="把文本写入工作区文件（覆盖或追加）。属于写操作，执行前需要用户确认。",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目标文件路径"},
                "content": {"type": "string", "description": "要写入的文本内容"},
                "mode": {"type": "string", "description": "overwrite（默认）或 append"},
            },
            "required": ["path", "content"],
        },
        handler=fs_write_file,
        requires_confirm=True,
        category="file",
    ))
    reg.register(ToolSpec(
        name="fs_analyze_file",
        description="统计文本文件：行数、字符数、中文关键词 Top10、JSON 结构摘要",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径"}},
            "required": ["path"],
        },
        handler=fs_analyze_file,
        category="file",
    ))
    reg.register(ToolSpec(
        name="todo_add",
        description="新增一条待办事项，持久化到 SQLite",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "待办标题"},
                "note": {"type": "string", "description": "备注"},
                "due": {"type": "string", "description": "截止时间，如 2026-10-28 18:00"},
                "priority": {"type": "string", "description": "low / medium / high，默认 medium"},
            },
            "required": ["title"],
        },
        handler=todo_add,
        category="todo",
    ))
    reg.register(ToolSpec(
        name="todo_list",
        description="查询待办清单，默认只列未完成的",
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "open（默认）/ done / all"},
            },
            "required": [],
        },
        handler=todo_list,
        category="todo",
    ))
    reg.register(ToolSpec(
        name="todo_done",
        description="把某条待办标记为已完成",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "待办 ID"}},
            "required": ["id"],
        },
        handler=todo_done,
        category="todo",
    ))
    reg.register(ToolSpec(
        name="todo_remove",
        description="删除某条待办。属于破坏性操作，执行前需要用户确认。",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "待办 ID"}},
            "required": ["id"],
        },
        handler=todo_remove,
        requires_confirm=True,
        category="todo",
    ))
    reg.register(ToolSpec(
        name="weather_get",
        description="查询城市当前天气与未来几天的预报（Open-Meteo，免密钥）",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，支持中文，如 北京 / Shanghai"},
                "days": {"type": "integer", "description": "预报天数 1-7，默认 3"},
            },
            "required": ["city"],
        },
        handler=weather_get,
        category="net",
    ))
    reg.register(ToolSpec(
        name="http_get_json",
        description="抓取一个 https 接口的内容（自动尝试 JSON 解析，响应截断到 2000 字符）",
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "必须是 https:// 开头"}},
            "required": ["url"],
        },
        handler=http_get_json,
        category="net",
    ))
    reg.register(ToolSpec(
        name="time_now",
        description="获取当前本地时间与星期几",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=time_now,
        category="net",
    ))
    reg.register(ToolSpec(
        name="code_run_python",
        description="执行一小段受限 Python 代码做计算或数据处理（默认关闭，需人工确认）",
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python 代码片段，不能 import，不能访问文件系统"},
            },
            "required": ["code"],
        },
        handler=code_run_python,
        requires_confirm=True,
        category="code",
    ))
    return reg


class McpServer:
    def __init__(self) -> None:
        self.registry = build_registry()
        self.tool_timeout = float(load_config().tool_timeout_sec)

    async def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """返回 None 表示这是一条通知（无需回包）。"""
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            return P.make_response(req_id, {
                "protocolVersion": P.MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })

        if method == "notifications/initialized" or (method or "").startswith("notifications/"):
            return None

        if method == "ping":
            return P.make_response(req_id, {})

        if method == "tools/list":
            return P.make_response(req_id, {"tools": self.registry.list_mcp_tools()})

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            try:
                result = await self.registry.call(name, arguments, timeout_sec=self.tool_timeout)
                text = json.dumps(result, ensure_ascii=False, default=str)
                return P.make_response(req_id, {
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                })
            except KeyError:
                return P.make_error(req_id, P.TOOL_NOT_FOUND, f"未知工具: {name}")
            except TimeoutError as exc:
                # 超时可重试，但仍如实告知模型，由模型决定是否换个参数重试
                return P.make_response(req_id, {
                    "content": [{"type": "text", "text": f"[工具超时] {exc}"}],
                    "isError": True,
                })
            except AgentError as exc:
                return P.make_response(req_id, {
                    "content": [{"type": "text", "text": f"[{exc.code}] {exc}"}],
                    "isError": True,
                })
            except Exception as exc:
                _LOG.exception("tool crashed: %s", name)
                return P.make_response(req_id, {
                    "content": [{"type": "text", "text": f"[internal_error] {type(exc).__name__}: {exc}"}],
                    "isError": True,
                })

        if req_id is None:
            return None
        return P.make_error(req_id, P.METHOD_NOT_FOUND, f"不支持的方法: {method}")


async def serve() -> None:
    """主循环：逐行读 stdin，处理后写 stdout。"""
    server = McpServer()
    loop = asyncio.get_running_loop()
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        raw = await loop.run_in_executor(None, stdin.readline)
        if not raw:
            _LOG.info("stdin closed, server exiting")
            break
        try:
            msg = P.decode(raw)
        except Exception:
            stdout.write(P.encode(P.make_error(None, P.PARSE_ERROR, "无法解析的 JSON-RPC 消息")))
            stdout.flush()
            continue

        try:
            resp = await server.handle(msg)
        except Exception:
            _LOG.error("handler crashed:\n%s", traceback.format_exc())
            resp = P.make_error(msg.get("id"), P.INTERNAL_ERROR, "服务端内部错误")

        if resp is not None:
            stdout.write(P.encode(resp))
            stdout.flush()


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.log_level, cfg.log_dir, stream=sys.stderr)
    init_db()
    # 强制 UTF-8 + \n 换行：Windows 默认的 cp936/gbk 会让中文工具名与输出乱码
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", newline="\n")  # type: ignore[attr-defined]
        except Exception:
            pass
    _LOG.info("mcp server starting, workspace=%s", cfg.workspace_root)
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
