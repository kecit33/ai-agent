"""编排层（Controller）：把 LLM、工具、记忆串成一条有界自主的循环。

这是整个 Agent 的"缰绳"。没有上限的 while True 是留给生产事故的，所以这里强制实现：
- max_steps：最多多少轮模型决策
- token 预算：累计 token 超上限即停
- 墙钟上限：整体耗时上限
- 连续工具失败次数上限
- 危险工具的 human-in-the-loop 确认点
- 终止时必给结论（调用一次不带工具的收尾，避免半截回答）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..common.config import Config, load_config
from ..common.errors import LlmUnavailable
from ..common.logging_setup import get_logger, new_trace_id
from ..common.retry import retry_async
from .llm import AssistantTurn, MockLlm, build_llm
from .mcp_client import McpStdioClient, ToolResult
from .memory import MemoryStore
from .prompts import PROMPT_VERSION, build_system_prompt

_LOG = get_logger("controller")

# 停止原因（日志与前端都会展示，便于定位"为什么它停了"）
STOP_FINAL = "final_answer"
STOP_MAX_STEPS = "max_steps"
STOP_WALL_CLOCK = "wall_clock"
STOP_TOKEN_BUDGET = "token_budget"
STOP_TOOL_FAILURES = "tool_failures"
STOP_ERROR = "error"

STOP_REASON_CN = {
    STOP_FINAL: "模型给出最终答复",
    STOP_MAX_STEPS: f"达到最大步数上限",
    STOP_WALL_CLOCK: "达到墙钟时间上限",
    STOP_TOKEN_BUDGET: "达到 token 预算上限",
    STOP_TOOL_FAILURES: "工具连续失败次数过多",
    STOP_ERROR: "执行过程发生错误",
}

ConfirmCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    confirmed: bool | None      # None = 该工具无需确认
    duration_ms: int
    output_preview: str


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_ms: int = 0
    stop_reason: str = STOP_FINAL
    trace_id: str = ""
    degraded: bool = False      # 是否走过降级路径（如 LLM 不可用切 Mock）


class AgentController:
    def __init__(
        self,
        cfg: Config | None = None,
        llm=None,
        mcp: McpStdioClient | None = None,
        memory: MemoryStore | None = None,
        confirm_cb: ConfirmCallback | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.llm = llm or build_llm(self.cfg)
        self.mcp = mcp or McpStdioClient(self.cfg)
        self.memory = memory or MemoryStore(self.cfg)
        self.confirm_cb = confirm_cb
        self.tools: list[dict[str, Any]] = []
        self.confirm_required: set[str] = set()
        self.trace_id = ""

    # ---------------- 生命周期 ----------------

    async def start(self) -> None:
        await self.mcp.start()
        await self._load_tools()

    async def _load_tools(self) -> None:
        mcp_tools = await self.mcp.list_tools()
        functions: list[dict[str, Any]] = []
        for t in mcp_tools:
            schema = dict(t.get("inputSchema") or {})
            requires_confirm = bool(schema.pop("x-requires-confirm", False))
            if requires_confirm:
                self.confirm_required.add(t["name"])
            functions.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": schema,
                },
            })
        self.tools = functions
        _LOG.info("loaded %d tools, confirm-required: %s", len(functions), sorted(self.confirm_required))

    async def aclose(self) -> None:
        await self.mcp.aclose()

    # ---------------- 主循环 ----------------

    async def run(self, user_text: str, session_id: int) -> AgentReply:
        self.trace_id = new_trace_id()
        started = time.monotonic()
        tokens_used = 0
        tool_failures = 0
        records: list[ToolCallRecord] = []
        degraded = False
        stop_reason = STOP_FINAL
        final_text = ""

        self.memory.append_message(session_id, "user", user_text)

        for step in range(1, self.cfg.max_steps + 1):
            elapsed = time.monotonic() - started
            if elapsed > self.cfg.max_wall_clock_sec:
                stop_reason = STOP_WALL_CLOCK
                break
            if tokens_used > self.cfg.max_token_budget:
                stop_reason = STOP_TOKEN_BUDGET
                break

            system_prompt = build_system_prompt(
                now=time.strftime("%Y-%m-%d %H:%M:%S"),
                facts_block=self.memory.render_facts(),
            )
            messages = await self.memory.build_context(session_id, system_prompt, llm=self.llm)

            try:
                turn = await self._chat_with_fallback(messages)
            except LlmUnavailable as exc:
                _LOG.error("LLM 不可用: %s", exc)
                if not isinstance(self.llm, MockLlm):
                    # 降级：切到规则模型，保证服务不中断（同时如实告知用户）
                    self.llm = MockLlm(self.cfg)
                    degraded = True
                    turn = await self.llm.chat(messages, tools=self.tools)
                else:
                    stop_reason = STOP_ERROR
                    final_text = f"模型不可用，且已无可用降级方案：{exc}"
                    break

            tokens_used += turn.prompt_tokens + turn.completion_tokens
            _LOG.info("step=%d finish=%s tokens=%d(+%d) trace=%s",
                      step, turn.finish_reason, tokens_used,
                      turn.prompt_tokens + turn.completion_tokens, self.trace_id)

            self.memory.append_message(
                session_id, "assistant", turn.content,
                tool_calls=[{"id": c["id"], "name": c["name"], "arguments": c["arguments"]}
                            for c in turn.tool_calls],
            )

            # 停止条件 1：模型没要工具，说明这是最终答复
            if not turn.tool_calls:
                final_text = turn.content
                stop_reason = STOP_FINAL
                break

            # 执行本轮全部工具调用
            for call in turn.tool_calls:
                name = call["name"]
                args = call.get("arguments") or {}
                began = time.monotonic()
                confirmed: bool | None = None

                if name in self.confirm_required and self.cfg.require_confirm:
                    confirmed = await self._ask_confirm(name, args)
                    if not confirmed:
                        result = ToolResult(
                            text=f"[用户拒绝] 已取消执行 {name}。若确实需要，请让用户重新确认。",
                            is_error=True,
                        )
                        tool_failures += 1
                        self._append_tool_result(session_id, call["id"], result)
                        records.append(ToolCallRecord(name, args, False, confirmed, 0, "用户拒绝"))
                        continue

                try:
                    result = await self.mcp.call_tool(name, args)
                except Exception as exc:
                    _LOG.error("tool %s raised: %s", name, exc)
                    result = ToolResult(text=f"[调用失败] {type(exc).__name__}: {exc}", is_error=True)

                if result.is_error:
                    tool_failures += 1
                duration_ms = int((time.monotonic() - began) * 1000)
                self._append_tool_result(session_id, call["id"], result)
                records.append(ToolCallRecord(
                    name=name,
                    arguments=args,
                    ok=not result.is_error,
                    confirmed=confirmed,
                    duration_ms=duration_ms,
                    output_preview=result.text[:300],
                ))

            if tool_failures >= 3:
                stop_reason = STOP_TOOL_FAILURES
                break
        else:
            stop_reason = STOP_MAX_STEPS

        # 非正常结束时必须给一句人话，不能把半截状态丢给用户
        if stop_reason != STOP_FINAL and not final_text:
            final_text = await self._wrap_up(session_id, stop_reason)

        wall_ms = int((time.monotonic() - started) * 1000)
        reply = AgentReply(
            text=final_text,
            tool_calls=records,
            steps=step_count(records, stop_reason),
            prompt_tokens=tokens_used,
            wall_ms=wall_ms,
            stop_reason=stop_reason,
            trace_id=self.trace_id,
            degraded=degraded,
        )
        _LOG.info("reply ready trace=%s stop=%s tools=%d wall_ms=%d",
                  self.trace_id, stop_reason, len(records), wall_ms)
        return reply

    async def _chat_with_fallback(self, messages: list[dict[str, Any]]) -> AssistantTurn:
        """LLM 调用统一入口：网络类异常退避重试，其它异常直接抛。"""
        return await retry_async(
            lambda: self.llm.chat(messages, tools=self.tools),
            attempts=2,
            base=1.0,
            cap=4.0,
            retry_on=(LlmUnavailable,),
            on_retry=lambda i, d, e: _LOG.warning("controller 重试 LLM 第 %d 次: %s", i, e),
        )

    def _append_tool_result(self, session_id: int, tool_call_id: str, result: ToolResult) -> None:
        self.memory.append_message(
            session_id, "tool", result.text, tool_call_id=tool_call_id
        )

    async def _ask_confirm(self, name: str, args: dict[str, Any]) -> bool:
        """危险操作的二次确认。没有回调时一律拒绝（默认安全，而不是默认放行）。"""
        if self.confirm_cb is None:
            _LOG.warning("工具 %s 需要确认但宿主未提供确认回调，已自动拒绝", name)
            return False
        try:
            return bool(await self.confirm_cb(name, args))
        except Exception as exc:
            _LOG.error("确认回调异常，按拒绝处理: %s", exc)
            return False

    async def _wrap_up(self, session_id: int, stop_reason: str) -> str:
        """被预算/失败打断时，不带工具再问一次，让模型基于已有信息给出结论。"""
        system_prompt = build_system_prompt(
            now=time.strftime("%Y-%m-%d %H:%M:%S"),
            facts_block=self.memory.render_facts(),
        )
        messages = await self.memory.build_context(session_id, system_prompt, llm=None)
        messages.append({
            "role": "system",
            "content": (
                f"你已触发停止条件：{STOP_REASON_CN.get(stop_reason, stop_reason)}。"
                "请基于已有信息给出简短结论，说明已完成到哪一步、还差什么，不要再调用工具。"
            ),
        })
        try:
            turn = await self.llm.chat(messages, tools=None)
            return turn.content or f"（已停止：{STOP_REASON_CN.get(stop_reason, stop_reason)}）"
        except Exception as exc:
            _LOG.warning("收尾总结失败: %s", exc)
            return f"（已停止：{STOP_REASON_CN.get(stop_reason, stop_reason)}）"


def step_count(records: list[ToolCallRecord], stop_reason: str) -> int:
    """步数用"工具调用轮次"近似：每次执行工具算一步，纯答复算 1 步。"""
    return max(1, len(records))


def reply_to_dict(reply: AgentReply) -> dict[str, Any]:
    """供 Web 前端消费的序列化格式。"""
    return {
        "text": reply.text,
        "steps": reply.steps,
        "stop_reason": reply.stop_reason,
        "stop_reason_cn": STOP_REASON_CN.get(reply.stop_reason, reply.stop_reason),
        "wall_ms": reply.wall_ms,
        "prompt_tokens": reply.prompt_tokens,
        "trace_id": reply.trace_id,
        "degraded": reply.degraded,
        "tool_calls": [
            {
                "name": r.name,
                "arguments": r.arguments,
                "ok": r.ok,
                "confirmed": r.confirmed,
                "duration_ms": r.duration_ms,
                "output_preview": r.output_preview,
            }
            for r in reply.tool_calls
        ],
    }
