"""记忆层：短期对话（SQLite 持久化）+ 长期事实（用户偏好）+ 上下文压缩。

三层记忆的分工：
- 短期：最近 N 条原始消息，保证多轮指代（"那上海呢"）能解析；
- 摘要：超出窗口的历史压缩成一段话，避免 token 无限膨胀；
- 长期：稳定事实（常驻城市、称呼、习惯）单独存表，并注入系统提示词。
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from ..common.config import Config, load_config
from ..common.logging_setup import get_logger

_LOG = get_logger("memory")

# 超过这个条数就触发摘要压缩
COMPRESS_THRESHOLD = 30
# 压缩后保留的原始消息条数
KEEP_RECENT = 12


@dataclass
class Session:
    id: int
    title: str
    created_at: float
    summary: str


class MemoryStore:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.db_path = self.cfg.db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT DEFAULT '',
                    tool_calls TEXT DEFAULT '',
                    tool_call_id TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    # ---------------- 会话 ----------------

    def create_session(self, title: str = "") -> int:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (title, summary, created_at, updated_at) VALUES (?,?,?,?)",
                (title, "", now, now),
            )
            return int(cur.lastrowid)

    def get_session(self, session_id: int) -> Session | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            return None
        return Session(id=row["id"], title=row["title"], created_at=row["created_at"], summary=row["summary"])

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"] or "(未命名会话)",
                "updated_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["updated_at"])),
            }
            for r in rows
        ]

    # ---------------- 消息 ----------------

    def append_message(
        self,
        session_id: int,
        role: str,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str = "",
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    session_id,
                    role,
                    content,
                    json.dumps(tool_calls or [], ensure_ascii=False),
                    tool_call_id,
                    now,
                ),
            )
            conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))

    def count_messages(self, session_id: int) -> int:
        with self._connect() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id=?", (session_id,)
            ).fetchone()["c"])

    def recent_messages(self, session_id: int, limit: int = KEEP_RECENT) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in reversed(rows):
            msg: dict[str, Any] = {"role": r["role"]}
            if r["content"]:
                msg["content"] = r["content"]
            if r["role"] == "assistant" and r["tool_calls"]:
                try:
                    calls = json.loads(r["tool_calls"])
                except json.JSONDecodeError:
                    calls = []
                if calls:
                    msg["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False),
                            },
                        }
                        for c in calls
                    ]
            if r["role"] == "tool" and r["tool_call_id"]:
                msg["tool_call_id"] = r["tool_call_id"]
            # OpenAI 协议要求 assistant 消息即使只带 tool_calls，也要有 content 字段（可为空串）
            if r["role"] == "assistant" and "content" not in msg:
                msg["content"] = ""
            out.append(msg)
        return out

    # ---------------- 上下文压缩 ----------------

    async def build_context(self, session_id: int, system_prompt: str, llm=None) -> list[dict[str, Any]]:
        """组装送入模型的消息列表：系统提示 + 历史摘要 + 最近原文。"""
        total = self.count_messages(session_id)
        session = self.get_session(session_id)
        summary = (session.summary if session else "") or ""

        if total > COMPRESS_THRESHOLD and llm is not None and not summary:
            summary = await self.compress(session_id, llm, keep=KEEP_RECENT)

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if summary:
            messages.append({
                "role": "system",
                "content": f"【更早对话的压缩摘要】\n{summary}\n（摘要可能丢失细节，涉及具体数字时以工具返回为准）",
            })
        messages.extend(self.recent_messages(session_id, KEEP_RECENT))
        return messages

    async def compress(self, session_id: int, llm, keep: int = KEEP_RECENT) -> str:
        """把较早的消息压缩成摘要。LLM 不可用时降级为"只保留最近 N 条"，绝不阻塞主流程。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC", (session_id,)
            ).fetchall()
        old = [r for r in rows[:-keep] if r["content"]]
        if not old:
            return ""

        transcript = "\n".join(f"{r['role']}: {r['content'][:300]}" for r in old)[-6000:]
        prompt = (
            "请把下面的对话压缩成 150 字以内的中文摘要，保留：用户目标、已确认的事实、"
            "已完成的动作、待办结论。不要添加原文没有的信息。\n\n" + transcript
        )
        try:
            turn = await llm.chat([{"role": "user", "content": prompt}], tools=None)
            summary = turn.content.strip()
        except Exception as exc:
            _LOG.warning("摘要生成失败，降级为截断策略: %s", exc)
            summary = f"（自动摘要失败，已丢弃 {len(old)} 条较早消息）"

        with self._connect() as conn:
            conn.execute("UPDATE sessions SET summary=? WHERE id=?", (summary, session_id))
        return summary

    # ---------------- 长期事实 ----------------

    def remember(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO facts (key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key.strip(), value.strip(), time.time()),
            )

    def forget(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM facts WHERE key=?", (key.strip(),))

    def all_facts(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM facts ORDER BY updated_at DESC").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def render_facts(self) -> str:
        facts = self.all_facts()
        if not facts:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in facts.items())
