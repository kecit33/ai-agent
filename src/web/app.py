"""Web 服务：把 CLI 的能力搬到浏览器里。

启动：
    python -m src.web.app            # 默认 http://127.0.0.1:8000
    uvicorn src.web.app:app --reload

危险操作确认采用"后台任务 + 轮询"模型：
    POST /api/chat -> 返回 task_id（不阻塞）
    GET  /api/task/{id} -> 查询状态，若遇到需要确认的工具会返回 confirm 信息
    POST /api/confirm -> 用户点允许/拒绝，唤醒后台任务继续执行
这样 HTTP 请求不会因为等待用户点击而长时间挂住。
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..agent.controller import AgentController, reply_to_dict
from ..agent.llm import build_llm
from ..agent.memory import MemoryStore
from ..common.config import Config, load_config
from ..common.logging_setup import get_logger, setup_logging

_LOG = get_logger("web")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 用户不点确认时最多等这么久，超时按拒绝处理（避免后台任务永久悬挂）
# 45 秒是一个"来得及读弹窗、又不会拖垮后续对话"的折中
CONFIRM_TIMEOUT_SEC = 45.0

# 后台任务的整体上限：controller 墙钟（默认 120s）+ 收尾/确认的余量
TASK_WATCHDOG_SEC = 200.0

# 同一会话的消息上下文必须串行读写（否则历史顺序会交错），
# 但不同会话之间没有理由互相排队。锁按 session_id 粒度管理。
LOCK_ACQUIRE_TIMEOUT_SEC = 3.0


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: int | None = None


class ConfirmRequest(BaseModel):
    task_id: str
    approved: bool


class AppState:
    def __init__(self) -> None:
        self.cfg: Config | None = None
        self.controller: AgentController | None = None
        self.memory: MemoryStore | None = None
        self.session_locks: dict[int, asyncio.Lock] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self.pending_confirm: dict[str, asyncio.Event] = {}
        self.confirm_result: dict[str, bool] = {}
        # 「自带 Key」模式：访问者在页面填自己的 DeepSeek Key（仅存其浏览器），
        # 每次请求经 X-LLM-Key 头带到服务端；这里按 Key 哈希缓存 LLM 实例。
        # Key 只驻留内存，不落盘、不进日志。
        self.default_llm: Any = None
        self.user_llms: dict[str, Any] = {}

    def lock_for(self, session_id: int) -> asyncio.Lock:
        lock = self.session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self.session_locks[session_id] = lock
        return lock


STATE = AppState()


def _get_user_llm(api_key: str, model: str = ""):
    """为自带 Key 的访问者构造/复用独立的真实模型实例。"""
    cache_key = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16] + "|" + (model or "")
    llm = STATE.user_llms.get(cache_key)
    if llm is None:
        assert STATE.cfg is not None
        overrides: dict[str, Any] = {"llm_mode": "openai", "llm_api_key": api_key}
        if model:
            overrides["llm_model"] = model
        llm = build_llm(dataclasses.replace(STATE.cfg, **overrides))
        STATE.user_llms[cache_key] = llm
    return llm


async def _confirm_cb_factory(task_id: str):
    async def _cb(name: str, arguments: dict[str, Any]) -> bool:
        task = STATE.tasks.setdefault(task_id, {})
        task["status"] = "awaiting_confirm"
        task["confirm"] = {"tool": name, "arguments": arguments}
        event = asyncio.Event()
        STATE.pending_confirm[task_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=CONFIRM_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            task["confirm"] = None
            return False
        return bool(STATE.confirm_result.pop(task_id, False))
    return _cb


def _busy_reply() -> dict[str, Any]:
    """同一会话的上一个任务还没结束（典型：确认弹窗没人点）时的明确反馈。"""
    return {
        "text": "上一条指令还在等待你的确认（或正在执行中）。\n"
                "如果看到确认弹窗，请先点「允许」或「拒绝」；"
                "如果弹窗已被关掉，稍等片刻它会自动按拒绝处理，然后再发一次即可。",
        "steps": 0,
        "stop_reason": "busy",
        "stop_reason_cn": "上一任务未完成",
        "wall_ms": 0,
        "prompt_tokens": 0,
        "trace_id": "",
        "degraded": False,
        "tool_calls": [],
    }


def _spawn_run(task_id: str, message: str, session_id: int, api_key: str = "", model: str = "") -> None:
    async def _job() -> None:
        assert STATE.controller is not None
        # 按请求选好模型实例，作为参数传入 run（不写回 controller.llm，
        # 否则两个带不同 Key 的并发请求会互相覆盖对方正在用的模型）
        llm = _get_user_llm(api_key, model) if api_key else STATE.default_llm
        lock = STATE.lock_for(session_id)

        # 拿锁带超时：同一会话的上一个任务还挂着时立刻给用户可操作的反馈，
        # 而不是让前端无声转圈直到对方超时（这是"问待办卡死"的根因之一）。
        try:
            await asyncio.wait_for(lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            STATE.tasks[task_id] = {
                "status": "done",
                "result": _busy_reply(),
                "session_id": session_id,
            }
            STATE.pending_confirm.pop(task_id, None)
            return

        try:
            # 确认回调必须在拿到锁之后再挂到 controller 上：
            # 提前挂会被仍在执行的上一个任务"借走"，出现 A 的确认弹到 B 头上的竞态。
            STATE.controller.confirm_cb = await _confirm_cb_factory(task_id)
            reply = await asyncio.wait_for(
                STATE.controller.run(message, session_id, llm=llm),
                timeout=TASK_WATCHDOG_SEC,  # 看门狗：任何意外挂起都不能让任务永远 running
            )
            STATE.tasks[task_id] = {
                "status": "done",
                "result": reply_to_dict(reply),
                "session_id": session_id,
            }
        except asyncio.TimeoutError:
            _LOG.error("task %s watchdog timeout (>%.0fs)", task_id, TASK_WATCHDOG_SEC)
            STATE.tasks[task_id] = {
                "status": "error",
                "error": f"任务超过 {TASK_WATCHDOG_SEC:.0f} 秒仍未完成，已被看门狗中止。请重试。",
                "session_id": session_id,
            }
        except Exception as exc:
            _LOG.exception("task %s failed", task_id)
            if api_key and ("401" in str(exc) or "Authentication" in str(exc)):
                # 自带 Key 鉴权失败：给出可操作的提示，而不是裸异常
                STATE.tasks[task_id] = {
                    "status": "done",
                    "result": {
                        "text": "你填写的 API Key 无效（DeepSeek 返回 401 鉴权失败）。\n"
                                "请检查 Key 是否正确、是否有余额；或清空侧边栏的 Key，改用内置演示模型。",
                        "steps": 0,
                        "stop_reason": "error",
                        "stop_reason_cn": "Key 鉴权失败",
                        "wall_ms": 0,
                        "prompt_tokens": 0,
                        "trace_id": "",
                        "degraded": False,
                        "tool_calls": [],
                    },
                    "session_id": session_id,
                }
            else:
                STATE.tasks[task_id] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "session_id": session_id,
                }
        finally:
            STATE.pending_confirm.pop(task_id, None)
            lock.release()

    asyncio.create_task(_job())


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    setup_logging(cfg.log_level, cfg.log_dir)
    STATE.cfg = cfg
    STATE.memory = MemoryStore(cfg)
    controller = AgentController(cfg=cfg, memory=STATE.memory, confirm_cb=None)
    await controller.start()
    # confirm_cb 需要 task_id，只能在请求进来时动态构造，这里通过 controller 属性替换
    STATE.controller = controller
    STATE.default_llm = controller.llm
    _LOG.info("web ready, tools=%d", len(controller.tools))
    try:
        yield
    finally:
        await controller.aclose()


app = FastAPI(title="小枢 Agent (MCP)", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/tools")
async def api_tools() -> dict[str, Any]:
    if STATE.controller is None:
        raise HTTPException(503, "Agent 尚未就绪")
    return {
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "requires_confirm": t["function"]["name"] in STATE.controller.confirm_required,
            }
            for t in STATE.controller.tools
        ],
        "model": getattr(STATE.controller.llm, "model", "unknown"),
        "prompt_version": None,
    }


@app.get("/api/sessions")
async def api_sessions() -> dict[str, Any]:
    assert STATE.memory is not None
    return {"sessions": STATE.memory.list_sessions()}


@app.post("/api/sessions")
async def api_new_session() -> dict[str, Any]:
    assert STATE.memory is not None
    sid = STATE.memory.create_session(title=time.strftime("网页会话 %m-%d %H:%M"))
    return {"session_id": sid}


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request) -> dict[str, Any]:
    if STATE.controller is None or STATE.memory is None:
        raise HTTPException(503, "Agent 尚未就绪")
    session_id = req.session_id or STATE.memory.create_session(
        title=req.message[:20] or time.strftime("网页会话 %m-%d %H:%M")
    )
    if STATE.memory.get_session(session_id) is None:
        raise HTTPException(404, f"会话 {session_id} 不存在")

    # 访问者自带的模型 Key（可选）：前端保存在其浏览器 localStorage，随请求头传入。
    # 服务端只用于本次任务的内存调用，不存储、不打印。
    api_key = (request.headers.get("x-llm-key") or "").strip()
    model = (request.headers.get("x-llm-model") or "").strip()

    task_id = uuid.uuid4().hex[:12]
    STATE.tasks[task_id] = {"status": "running", "session_id": session_id}
    # 确认回调在 _spawn_run 拿到会话锁之后再挂载（避免被仍在执行的上一个任务借走）
    _spawn_run(task_id, req.message, session_id, api_key, model)
    return {"task_id": task_id, "session_id": session_id, "status": "running"}


@app.get("/api/task/{task_id}")
async def api_task(task_id: str) -> dict[str, Any]:
    task = STATE.tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在或已过期")
    return {"task_id": task_id, **task}


@app.post("/api/confirm")
async def api_confirm(req: ConfirmRequest) -> dict[str, Any]:
    event = STATE.pending_confirm.get(req.task_id)
    if event is None:
        raise HTTPException(409, "当前没有等待确认的操作")
    STATE.confirm_result[req.task_id] = req.approved
    task = STATE.tasks.get(req.task_id, {})
    task["status"] = "running"
    task["confirm"] = None
    event.set()
    return {"ok": True}


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    return {
        "ok": STATE.controller is not None and STATE.controller.mcp.available,
        "model": getattr(STATE.controller.llm, "model", "unknown") if STATE.controller else "",
    }


def main() -> None:
    import os

    import uvicorn

    # 监听地址与端口均可通过环境变量覆盖：
    # - 本地开发默认只监听 127.0.0.1，避免暴露到局域网；
    # - 部署到线上时平台会注入 PORT，并需要绑定 0.0.0.0，因此用 WEB_HOST 显式放开。
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
