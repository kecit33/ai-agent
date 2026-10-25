"""文件类工具：列目录 / 读 / 写 / 统计分析。

安全设计（双保险）：
1. 路径沙箱——任何路径都先 resolve 再判断是否仍在 WORKSPACE_ROOT 之内，
   杜绝 ../../etc/passwd 这类越权读取；
2. 写操作标记 requires_confirm=True，由客户端在调用前请用户确认。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from ..common.config import load_config
from ..common.errors import PermissionDenied, UserInputError

_CFG = load_config()
WORKSPACE_ROOT = _CFG.workspace_root

# 单文件最多读取的字符数，防止一个 10MB 日志把上下文撑爆
MAX_READ_CHARS = 8000
MAX_WRITE_CHARS = 20000


def resolve_in_workspace(raw_path: str) -> Path:
    """把相对/绝对路径都规范到沙箱内，越界直接抛 PermissionDenied。

    注：模型常把沙箱目录名一起写进相对路径（如 "workspace/a.txt"），
    直接拼接会产生 workspace/workspace/a.txt 的幽灵路径，因此先剥掉
    多余的目录名前缀（真实模型演示中发现并修复的问题）。
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise UserInputError("path 必须是非空字符串")
    raw = raw_path.strip().replace("\\", "/")
    candidate = Path(raw)
    if not candidate.is_absolute():
        prefix = WORKSPACE_ROOT.name + "/"
        while raw.startswith(prefix):
            raw = raw[len(prefix):]
        candidate = WORKSPACE_ROOT / (raw or ".")
    resolved = candidate.resolve()
    root = WORKSPACE_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionDenied(f"路径越界，只允许访问 {root} 内的文件: {raw_path}")
    return resolved


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}TB"


async def fs_list_dir(args: dict) -> dict:
    rel = args.get("path", ".")
    recursive = bool(args.get("recursive", False))
    base = resolve_in_workspace(rel)
    if not base.exists():
        raise UserInputError(f"路径不存在: {rel}")
    if not base.is_dir():
        raise UserInputError(f"不是目录: {rel}")

    entries = []
    walker = base.rglob("*") if recursive else base.iterdir()
    limit = 200
    for p in walker:
        if len(entries) >= limit:
            entries.append({"path": "...", "note": f"结果超过 {limit} 条，已截断"})
            break
        try:
            stat = p.stat()
        except OSError:
            continue
        entries.append({
            "path": str(p.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "type": "dir" if p.is_dir() else "file",
            "size": _human_size(stat.st_size) if p.is_file() else "-",
        })
    return {"root": str(base), "count": len(entries), "entries": entries}


async def fs_read_file(args: dict) -> dict:
    path = resolve_in_workspace(args["path"])
    if not path.exists():
        raise UserInputError(f"文件不存在: {args['path']}")
    if not path.is_file():
        raise UserInputError(f"不是文件: {args['path']}")
    max_chars = int(args.get("max_chars", MAX_READ_CHARS))
    max_chars = max(200, min(max_chars, MAX_READ_CHARS))

    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > max_chars
    return {
        "path": str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "chars": len(text),
        "truncated": truncated,
        "content": text[:max_chars],
    }


async def fs_write_file(args: dict) -> dict:
    path = resolve_in_workspace(args["path"])
    content = args["content"]
    if not isinstance(content, str):
        raise UserInputError("content 必须是字符串")
    if len(content) > MAX_WRITE_CHARS:
        raise UserInputError(f"单次写入上限 {MAX_WRITE_CHARS} 字符，当前 {len(content)}")

    mode = args.get("mode", "overwrite")
    if mode not in ("overwrite", "append"):
        raise UserInputError("mode 只能是 overwrite 或 append")

    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append" and path.exists():
        with path.open("a", encoding="utf-8") as f:
            f.write("\n" + content)
    else:
        path.write_text(content, encoding="utf-8")
    return {
        "ok": True,
        "path": str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "written_chars": len(content),
        "mode": mode,
    }


async def fs_analyze_file(args: dict) -> dict:
    """对文本文件做轻量统计：行数、字符数、中文词频 Top10、JSON 结构摘要。"""
    path = resolve_in_workspace(args["path"])
    if not path.is_file():
        raise UserInputError(f"不是文件: {args['path']}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > 200_000:
        text = text[:200_000]

    lines = text.splitlines()
    cn_words = re.findall(r"[\u4e00-\u9fa5]{2,6}", text)
    top_cn = Counter(cn_words).most_common(10)

    result: dict = {
        "path": str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "lines": len(lines),
        "chars": len(text),
        "words_en": len(re.findall(r"[A-Za-z]+", text)),
        "top_keywords": [{"word": w, "count": c} for w, c in top_cn],
    }
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            obj = json.loads(stripped)
            result["json_type"] = type(obj).__name__
            if isinstance(obj, dict):
                result["json_keys"] = list(obj.keys())[:20]
            elif isinstance(obj, list):
                result["json_len"] = len(obj)
        except json.JSONDecodeError:
            result["json_type"] = "invalid_json"
    return result
