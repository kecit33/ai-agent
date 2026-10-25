"""MCP 协议层：JSON-RPC 2.0 消息封装与标准错误码。

为什么自己实现协议而不直接用官方 SDK：
1. 教学可读性——作业要讲清"客户端和服务端到底说了什么"，SDK 会把 initialize /
   notifications/initialized / tools/list / tools/call 这套握手藏进黑盒；
2. 依赖可控——官方 SDK 迭代很快，自研约 150 行即可覆盖三个核心方法。

协议要点（stdio 传输）：
- 每条消息是一行 JSON（newline-delimited JSON），以 \\n 结尾；
- stdout 只能写协议消息，日志必须走 stderr，否则协议会被日志打乱；
- 握手顺序：client 发 initialize -> server 回 result -> client 发 notifications/initialized（通知无 id，不回包）。
"""
from __future__ import annotations

import json
from typing import Any

JSONRPC_VERSION = "2.0"

# 协议版本协商字段。若服务端要求更高版本，改这里即可。
# [待核实: 以 https://modelcontextprotocol.io/specification 的 versioning 章节为准]
MCP_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC 2.0 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# 服务端自定义错误码（JSON-RPC 规定 -32000 起由实现自行定义）
TOOL_EXECUTION_ERROR = -32001
TOOL_NOT_FOUND = -32002


def make_request(req_id: int | str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def make_notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """通知：单向消息，没有 id，接收方不应回包。"""
    msg: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def make_response(req_id: int | str | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}


def make_error(req_id: int | str | None, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "error": err}


def encode(obj: dict[str, Any]) -> bytes:
    """序列化为一行 UTF-8 JSON（含换行符）。ensure_ascii=False 保证中文不被转义。"""
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("json-rpc message must be an object")
    return obj
