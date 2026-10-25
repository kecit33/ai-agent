"""把项目源码汇总成单文件 docs/代码汇总.md，便于通读、打印与提交检查。

用法（在项目根目录执行）：
    python tools/gen_code_doc.py
    python tools/gen_code_doc.py --output docs/代码汇总.md

设计取舍：
- 只收"代码与配置"，不收 prose 文档（README、blog），避免文件过大；
- 排除虚拟环境、运行时数据、缓存与 Git 元数据；
- 每个文件带行数统计与围栏语言标注，长文件不截断。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 汇总顺序：先配置，再按 公共层 → MCP Server → Agent → Web → 语音 → 测试 → 评测 → 示例数据
DIR_PRIORITY = [
    "requirements.txt",
    "requirements-voice.txt",
    ".env.example",
    ".gitignore",
    "src/common/",
    "src/mcp_server/",
    "src/agent/",
    "src/web/",
    "src/voice/",
    "tests/",
    "evals/",
    "workspace/",
]

EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".data", ".idea", ".vscode", "node_modules"}
EXCLUDE_FILES = {"agent.db", "agent.log"}
# 安全红线：含真实密钥的环境文件绝不进汇总（.env 只在 .env.example 中体现结构）
EXCLUDE_REL = {".env", ".env.local", "README.md", "evals/baseline.md", "evals/demo_results.md"}

FENCE_BY_SUFFIX = {
    ".py": "python",
    ".md": "markdown",
    ".html": "html",
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".txt": "text",
    ".cfg": "ini",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
}
# 无后缀或特殊文件名的围栏语言
FENCE_BY_NAME = {
    ".env.example": "ini",
    ".gitignore": "text",
    "requirements.txt": "text",
    "requirements-voice.txt": "text",
}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    if path.name in EXCLUDE_FILES:
        return True
    if rel.as_posix() in EXCLUDE_REL:
        return True
    if path.suffix == ".pyc":
        return True
    return False


def collect_files() -> list[Path]:
    """按 DIR_PRIORITY 的顺序收集文件，未列出的目录排在最后。"""
    buckets: dict[str, list[Path]] = {key: [] for key in DIR_PRIORITY}
    others: list[Path] = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel.startswith("tools/") or rel.startswith("docs/") or rel.startswith("blog/"):
            continue
        matched = None
        for key in DIR_PRIORITY:
            if rel == key or rel.startswith(key if key.endswith("/") else key + "/"):
                matched = key
                break
        if matched:
            buckets[matched].append(path)
        else:
            others.append(path)

    result: list[Path] = []
    for key in DIR_PRIORITY:
        result.extend(sorted(buckets[key], key=lambda p: p.as_posix()))
    result.extend(sorted(others, key=lambda p: p.as_posix()))
    return result


def fence_for(path: Path) -> str:
    if path.name in FENCE_BY_NAME:
        return FENCE_BY_NAME[path.name]
    return FENCE_BY_SUFFIX.get(path.suffix, "")


def build_tree(files: list[Path]) -> str:
    """把文件清单折叠成一棵标准目录树（目录只出现一次）。"""
    tree: dict = {}
    for path in files:
        node = tree
        parts = path.relative_to(PROJECT_ROOT).parts
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = None

    lines: list[str] = []

    def walk(node: dict, prefix: str = "") -> None:
        keys = sorted(node.keys(), key=lambda k: (k.endswith("/"), k))
        for i, key in enumerate(keys):
            last = i == len(keys) - 1
            connector = "└── " if last else "├── "
            lines.append(prefix + connector + key.rstrip("/") + ("" if key.endswith("/") else ""))
            child = node[key]
            if child:
                walk(child, prefix + ("    " if last else "│   "))

    walk(tree)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 docs/代码汇总.md")
    parser.add_argument("--output", default="docs/代码汇总.md", help="输出文件路径（相对项目根目录）")
    args = parser.parse_args()

    files = collect_files()
    total_lines = 0
    chunks: list[str] = []

    chunks.append("# mcp-agent 代码汇总\n")
    chunks.append(
        f"> 由 `tools/gen_code_doc.py` 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}。"
        "改代码后重新执行 `python tools/gen_code_doc.py` 即可刷新本文件。\n"
    )
    chunks.append(f"> 共收录 **{len(files)} 个文件**（仅代码与配置，不含 README 与博文）。\n")
    chunks.append("\n---\n\n## 一、文件清单\n\n```\n" + build_tree(files) + "\n```\n")

    chunks.append("\n---\n\n## 二、源码全文\n")
    for path in files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            chunks.append(f"\n### `{rel}`\n\n> 二进制或非 UTF-8 文件，已跳过内容。\n")
            continue
        line_count = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        total_lines += line_count
        fence = fence_for(path)
        chunks.append(f"\n### `{rel}`\n")
        chunks.append(f"\n> {line_count} 行\n\n")
        chunks.append(f"```{fence}\n{content.rstrip()}\n```\n")

    chunks.append(f"\n---\n\n*全文合计约 {total_lines} 行。*\n")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(chunks), encoding="utf-8")
    print(f"已生成：{out_path}")
    print(f"收录文件 {len(files)} 个，约 {total_lines} 行。")


if __name__ == "__main__":
    main()
