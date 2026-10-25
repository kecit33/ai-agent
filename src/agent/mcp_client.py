"""MCP 客户端：拉起 MCP Server 子进程，完成握手并调用工具。

它是 Agent 的"手"——LLM 只负责决定用哪个工具，真正干活的是这里。
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any

from ..common.config import PROJECT_ROOT, Config, load_config
from ..common.errors import FatalError, RetryableError, UserInputError
from ..common.logging_setup import get_logger
from ..common.retry import retry_async
from ..mcp_server import protocol as P

_LOG = get_logger("mcp_client")


class McpConnectionError(FatalError):
    """子进程没起来 / 握手失败 / 进程挂掉——属于配置或环境问题，重试没意义。"""
    code = "mcp_connection_failed"


@dataclass
class ToolResult:
    text: str
    is_error: bool

    def as_tool_message(self, tool_call_id: str) -> dict[str, Any]:
        """转成 OpenAI 协议里"工具执行结果"这条消息。"""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": self.text}


class McpStdioClient:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 0
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self.server_info: dict[str, Any] = {}
        self.available = False

    # ---------------- 生命周期 ----------------

    async def start(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        try:
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "src.mcp_server.server",
                cwd=str(PROJECT_ROOT),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            raise McpConnectionError(f"无法启动 MCP Server 子进程: {exc}") from exc

        self._reader_task = asyncio.create_task(self._read_stdout_loop())
        self._stderr_task = asyncio.create_task(self._read_stderr_loop())

        try:
            await self._initialize()
        except Exception as exc:
            await self.aclose()
            raise McpConnectionError(f"MCP 握手失败: {exc}") from exc

        self.available = True
        _LOG.info("mcp connected: %s", self.server_info.get("serverInfo"))

    async def _initialize(self) -> None:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": P.MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "homework-agent", "version": "1.0.0"},
            },
            timeout_sec=20.0,
        )
        self.server_info = result
        self.send_notification(P.make_notification("notifications/initialized"))

    async def aclose(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(McpConnectionError("MCP 连接已关闭"))
        self._pending.clear()
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.proc.kill()
        self.available = False

    # ---------------- 协议收发 ----------------

    def _alloc_id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def request(self, method: str, params: dict[str, Any], timeout_sec: float = 20.0) -> Any:
        if self.proc is None or self.proc.stdin is None:
            raise McpConnectionError("MCP 子进程未启动")
        req_id = self._alloc_id()
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        payload = P.encode(P.make_request(req_id, method, params))
        self.proc.stdin.write(payload)
        await self.proc.stdin.drain()
        try:
            msg = await asyncio.wait_for(fut, timeout=timeout_sec)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise RetryableError(f"MCP 请求 {method} 超时（>{timeout_sec}s）")
        if "error" in msg:
            err = msg["error"]
            code = err.get("code")
            # 工具名不存在 = 模型幻觉：这不是网络问题，重试没用，应把错误回喂给模型让它改
            if code == P.TOOL_NOT_FOUND:
                raise UserInputError(f"[{code}] {err.get('message')}")
            raise RetryableError(f"MCP 错误 {method}: [{code}] {err.get('message')}")
        return msg.get("result")

    def send_notification(self, msg: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        self.proc.stdin.write(P.encode(msg))

    async def _read_stdout_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = P.decode(line)
                except Exception:
                    _LOG.warning("收到无法解析的消息: %r", line[:200])
                    continue
                rid = msg.get("id")
                fut = self._pending.pop(rid, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
        except asyncio.CancelledError:
            return
        except Exception:
            _LOG.exception("stdout 读取循环异常退出")
        finally:
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(McpConnectionError("MCP 连接中断"))
            self._pending.clear()
            self.available = False

    async def _read_stderr_loop(self) -> None:
        """把服务端的日志转发到本进程的日志系统，避免子进程 stderr 缓冲区写满卡死。"""
        assert self.proc is not None and self.proc.stderr is not None
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    _LOG.debug("server: %s", text[:500])
        except asyncio.CancelledError:
            return
        except Exception:
            return

    # ---------------- 对外能力 ----------------

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self.request("tools/list", {}, timeout_sec=15.0)
        return list(result.get("tools", []))

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout_sec: float | None = None) -> ToolResult:
        """调用工具，带一次退避重试（网络类工具偶发抖动很常见）。"""
        timeout = timeout_sec or float(self.cfg.tool_timeout_sec)

        async def _once() -> ToolResult:
            if not self.available or self.proc is None:
                raise McpConnectionError("MCP 不可用")
            try:
                result = await self.request(
                    "tools/call", {"name": name, "arguments": arguments}, timeout_sec=timeout
                )
            except UserInputError as exc:
                # 幻觉工具名 / 参数明显不合法：转成工具结果让模型看见，而不是抛给上层炸掉整个循环
                return ToolResult(text=f"[bad_request] {exc}", is_error=True)
            content = (result or {}).get("content") or []
            text = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
            return ToolResult(text=text or "(工具无输出)", is_error=bool(result.get("isError")))

        return await retry_async(
            _once,
            attempts=2,
            base=0.6,
            cap=3.0,
            retry_on=(RetryableError,),
            on_retry=lambda i, d, e: _LOG.warning("tool %s 第 %d 次重试，%.1fs 后执行: %s", name, i, d, e),
        )
