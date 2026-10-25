"""待办清单工具：把"说到"变成"做到"的最小闭环，数据落 SQLite，进程重启不丢。

表结构与 Agent 的记忆表共用同一个 sqlite 文件（.data/agent.db），但表名互不干扰。
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any

from ..common.config import load_config
from ..common.errors import UserInputError

_CFG = load_config()
DB_PATH = _CFG.db_path

VALID_STATUS = ("open", "done")
VALID_PRIORITY = ("low", "medium", "high")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL 让"Agent 主进程写消息表"与"MCP 子进程写待办表"可以并发，不互相阻塞
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT NOT NULL,
                note       TEXT DEFAULT '',
                due        TEXT DEFAULT '',
                priority   TEXT DEFAULT 'medium',
                status     TEXT DEFAULT 'open',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status)")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "note": row["note"],
        "due": row["due"],
        "priority": row["priority"],
        "status": row["status"],
        "created_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(row["created_at"])),
    }


async def todo_add(args: dict) -> dict:
    title = str(args["title"]).strip()
    if not title:
        raise UserInputError("title 不能为空")
    priority = str(args.get("priority", "medium")).lower()
    if priority not in VALID_PRIORITY:
        raise UserInputError(f"priority 只能是 {VALID_PRIORITY}")
    now = time.time()
    with _connect() as conn:
        # 去重：同一条待办被重复添加（模型或用户连发两次）会污染清单，
        # 这里按"标题 + 截止时间"判断，命中已存在的未完成任务则直接返回原条目。
        due = str(args.get("due", ""))
        dup = conn.execute(
            "SELECT * FROM todos WHERE title = ? AND due = ? AND status = 'open' ORDER BY id ASC LIMIT 1",
            (title, due),
        ).fetchone()
        if dup is not None:
            hit = _row_to_dict(dup)
            return {
                "ok": True,
                "id": hit["id"],
                "title": title,
                "duplicate": True,
                "message": f"已存在相同待办 #{hit['id']}，未重复添加：{title}",
            }

        cur = conn.execute(
            "INSERT INTO todos (title, note, due, priority, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (title, str(args.get("note", "")), due, priority, "open", now, now),
        )
        todo_id = int(cur.lastrowid)
    return {"ok": True, "id": todo_id, "title": title, "message": f"已添加待办 #{todo_id}: {title}"}


async def todo_list(args: dict) -> dict:
    status = str(args.get("status", "open")).lower()
    if status not in VALID_STATUS and status != "all":
        raise UserInputError(f"status 只能是 open / done / all")
    sql = "SELECT * FROM todos"
    params: tuple = ()
    if status != "all":
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, id ASC"
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    items = [_row_to_dict(r) for r in rows]
    return {"status": status, "count": len(items), "items": items}


async def todo_done(args: dict) -> dict:
    todo_id = int(args["id"])
    with _connect() as conn:
        row = conn.execute("SELECT id, title FROM todos WHERE id = ?", (todo_id,)).fetchone()
        if row is None:
            raise UserInputError(f"待办 #{todo_id} 不存在")
        conn.execute(
            "UPDATE todos SET status='done', updated_at=? WHERE id=?", (time.time(), todo_id)
        )
    return {"ok": True, "id": todo_id, "message": f"已完成待办 #{todo_id}: {row['title']}"}


async def todo_remove(args: dict) -> dict:
    todo_id = int(args["id"])
    with _connect() as conn:
        row = conn.execute("SELECT id, title FROM todos WHERE id = ?", (todo_id,)).fetchone()
        if row is None:
            raise UserInputError(f"待办 #{todo_id} 不存在")
        conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
    return {"ok": True, "id": todo_id, "message": f"已删除待办 #{todo_id}: {row['title']}"}
