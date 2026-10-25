"""代码执行工具（默认关闭，属高危能力）。

重要说明（请在作业答辩时如实陈述）：
本工具通过"裁剪 __builtins__ + 超时返回"降低风险，但 **Python 无法真正杀死一个跑飞的线程**，
因此它不是安全沙箱。生产环境必须使用容器 / gVisor / 独立低权限进程来隔离。
默认 ALLOW_CODE_EXEC=false，需显式开启，且执行前仍需人工二次确认。
"""
from __future__ import annotations

import io
import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor

from ..common.config import load_config
from ..common.errors import PermissionDenied, UserInputError

_CFG = load_config()
EXEC_TIMEOUT_SEC = 5.0
MAX_CODE_CHARS = 2000

# 允许的内置函数白名单：能算能格式化，但摸不到文件系统和进程
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int, "len": len,
    "list": list, "map": map, "max": max, "min": min, "pow": pow, "range": range,
    "repr": repr, "reversed": reversed, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip, "print": print,
}

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="code-exec")


def _run_snippet(code: str) -> dict:
    stdout = io.StringIO()
    sandbox_globals = {"__builtins__": dict(_SAFE_BUILTINS)}
    result_repr = None
    try:
        with contextlib.redirect_stdout(stdout):
            # 用 eval 取表达式值；语句类的副作用靠 print 输出
            try:
                value = eval(compile(code, "<agent_snippet>", "eval"), sandbox_globals, {})
                result_repr = repr(value)
            except SyntaxError:
                exec(compile(code, "<agent_snippet>", "exec"), sandbox_globals, {})
        return {"ok": True, "stdout": stdout.getvalue()[:2000], "result": result_repr, "error": None}
    except Exception as exc:  # 用户代码可能抛任何异常，必须兜住而不是让工具进程崩掉
        return {
            "ok": False,
            "stdout": stdout.getvalue()[:2000],
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def code_run_python(args: dict) -> dict:
    if not _CFG.allow_code_exec:
        raise PermissionDenied(
            "代码执行已关闭。设置 ALLOW_CODE_EXEC=true 可开启（仅建议在受控环境演示）"
        )
    code = args["code"]
    if not isinstance(code, str) or not code.strip():
        raise UserInputError("code 必须是非空字符串")
    if len(code) > MAX_CODE_CHARS:
        raise UserInputError(f"代码长度上限 {MAX_CODE_CHARS} 字符")

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_EXECUTOR, _run_snippet, code),
            timeout=EXEC_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        # 线程无法强制终止，只能立刻返回并告警；后续请求仍可能受该线程影响
        return {
            "ok": False,
            "stdout": "",
            "result": None,
            "error": f"执行超时（>{EXEC_TIMEOUT_SEC}s）。注意：Python 无法强制杀死线程，"
                     f"死循环代码会持续占用一个工作线程，请重启 MCP Server 恢复。",
        }
