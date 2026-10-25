"""可观测性基础：trace_id 贯穿 + 结构化日志 + PII / 密钥脱敏（C6）。

设计取舍：
- 用 contextvars 存 trace_id，异步并发下不会串号；
- 所有日志写 stderr 或文件，绝不能污染 stdout —— MCP stdio 传输依赖 stdout 传 JSON-RPC，
  一旦被日志写入就会协议错乱（这是自研 MCP Server 最常见的坑）。
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

# 脱敏规则：命中即用占位符替换，防止手机号/邮箱/身份证/密钥进入日志与上下文
_REDACT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), "sk-***REDACTED***"),
    (re.compile(r"(?i)(api[_-]?key|authorization|token)\s*[:=]\s*[\"']?([A-Za-z0-9._\-]{6,})"),
     r"\1=***REDACTED***"),
    (re.compile(r"1[3-9]\d{9}"), "1**********"),                       # 手机号
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "***@***.***"),           # 邮箱
    (re.compile(r"\b\d{17}[\dXx]\b"), "******************"),           # 身份证
]


def redact(text: Any) -> str:
    """对任意对象做脱敏后转字符串，供日志与前端展示共用。"""
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    for pattern, repl in _REDACT_RULES:
        s = pattern.sub(repl, s)
    return s


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:16]
    trace_id_var.set(tid)
    return tid


class TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """单行 JSON 日志，便于后续接 ELK / Loki 做检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "trace_id": getattr(record, "trace_id", "-"),
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", log_dir: Path | None = None, stream=sys.stderr) -> None:
    """初始化根日志器。stream 默认 stderr，避免干扰 MCP 的 stdout 通道。"""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = JsonFormatter()
    flt = TraceFilter()

    sh = logging.StreamHandler(stream)
    sh.setFormatter(fmt)
    sh.addFilter(flt)
    root.addHandler(sh)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "agent.log", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(flt)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
