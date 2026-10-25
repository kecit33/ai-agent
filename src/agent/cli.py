"""命令行入口：最小可用的人机交互界面，也是验收演示的主场地。

    python -m src.agent.cli                  # 交互式
    python -m src.agent.cli --once "北京天气"  # 单条指令后退出
    python -m src.agent.cli --voice          # 打开语音输入/输出（需装 requirements-voice.txt）
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

from ..common.config import load_config
from ..common.logging_setup import setup_logging
from .controller import AgentController, STOP_REASON_CN
from .llm import MockLlm
from .memory import MemoryStore

HELP = """可用命令：
  /help              显示本帮助
  /tools             列出 MCP 提供的工具
  /new               开启新会话
  /sessions          列出历史会话
  /use <id>          切换到指定会话
  /facts             查看长期记忆
  /remember k=v      记住一条长期事实，如 /remember 常驻城市=北京
  /forget k          忘掉一条长期事实
  /exit              退出
直接输入自然语言即可对话。"""


async def _async_input(prompt: str) -> str:
    """在事件循环里安全地使用阻塞式 input()。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


def _print_tool_trace(records: list[Any]) -> None:
    for r in records:
        flag = "✅" if r.ok else "❌"
        confirm = ""
        if r.confirmed is True:
            confirm = " [已确认]"
        elif r.confirmed is False:
            confirm = " [用户拒绝]"
        args = str(r.arguments)
        if len(args) > 120:
            args = args[:120] + "…"
        print(f"  {flag} {r.name}{confirm} ({r.duration_ms}ms) args={args}")
        if not r.ok:
            print(f"     ↳ {r.output_preview[:200]}")


async def run_cli(args: argparse.Namespace) -> int:
    cfg = load_config()
    setup_logging(cfg.log_level, cfg.log_dir, stream=sys.stderr)

    memory = MemoryStore(cfg)

    async def confirm_cb(name: str, arguments: dict[str, Any]) -> bool:
        print(f"\n⚠️  危险操作确认：即将执行 {name}，参数 {arguments}")
        ans = await _async_input("  允许执行？(y/N): ")
        return ans.strip().lower() in ("y", "yes")

    controller = AgentController(cfg=cfg, memory=memory, confirm_cb=confirm_cb)
    await controller.start()

    model_name = getattr(controller.llm, "model", "unknown")
    is_mock = isinstance(controller.llm, MockLlm)
    print("=" * 68)
    print("  小枢 · 能说会做的 AI 智能体（MCP 版）")
    print(f"  模型: {model_name}" + ("  ⚠️ Mock 规则模型，仅供无 Key 演示" if is_mock else ""))
    print(f"  工具: {len(controller.tools)} 个，来自独立 MCP Server 子进程")
    print(f"  预算: max_steps={cfg.max_steps} wall_clock={cfg.max_wall_clock_sec}s "
          f"token={cfg.max_token_budget}")
    print("  输入 /help 查看命令")
    print("=" * 68)

    voice_enabled = args.voice
    stt = tts = None
    if voice_enabled:
        try:
            from ..voice.stt import listen_once
            from ..voice.tts import speak
            stt, tts = listen_once, speak
            print("🎙️  语音模式已开启（回车开始说话）")
        except Exception as exc:
            print(f"⚠️  语音模块不可用，已退回文本模式：{exc}")
            voice_enabled = False

    session_id = args.session if args.session else memory.create_session(
        title=time.strftime("会话 %m-%d %H:%M")
    )
    args.session = session_id  # 让 /new、/use 能直接改写当前会话
    print(f"会话 ID: {session_id}\n")

    try:
        while True:
            if args.once:
                user_text = args.once
            else:
                try:
                    if voice_enabled:
                        user_text = await _async_input("\n🎙️  按回车开始录音（或直接打字）> ")
                        if not user_text.strip():
                            user_text = await stt()
                            print(f"你说：{user_text}")
                        if not user_text.strip():
                            continue
                    else:
                        user_text = await _async_input("\n你 > ")
                except (EOFError, KeyboardInterrupt):
                    print("\n再见。")
                    break

            user_text = user_text.strip()
            if not user_text:
                continue
            if user_text.startswith("/"):
                if user_text in ("/exit", "/quit"):
                    print("再见。")
                    break
                await _handle_command(user_text, controller, memory, args)
                session_id = args.session  # /new、/use 会改写 args.session
                continue

            print()
            reply = await controller.run(user_text, session_id)
            _print_tool_trace(reply.tool_calls)
            if voice_enabled and tts is not None:
                tts(reply.text)
            print(f"\n小枢 > {reply.text}")
            print(f"\n[trace={reply.trace_id} 步数={reply.steps} 停止原因="
                  f"{STOP_REASON_CN.get(reply.stop_reason, reply.stop_reason)} "
                  f"耗时={reply.wall_ms}ms tokens≈{reply.prompt_tokens}"
                  f"{' 降级' if reply.degraded else ''}]")

            if args.once:
                break
    finally:
        await controller.aclose()
    return 0


async def _handle_command(
    cmd: str, controller: AgentController, memory: MemoryStore, args: argparse.Namespace
) -> bool:
    """处理斜杠命令；返回 True 表示已识别并处理。"""
    parts = cmd.split(maxsplit=1)
    name = parts[0]
    rest = parts[1].strip() if len(parts) > 1 else ""

    if name == "/help":
        print(HELP)
        return True
    if name == "/tools":
        for t in controller.tools:
            fn = t["function"]
            mark = " ⚠️需确认" if fn["name"] in controller.confirm_required else ""
            print(f"  · {fn['name']}{mark}: {fn['description']}")
        return True
    if name == "/new":
        args.session = memory.create_session(title=time.strftime("会话 %m-%d %H:%M"))
        print(f"已开启新会话，ID={args.session}")
        return True
    if name == "/sessions":
        for s in memory.list_sessions():
            print(f"  #{s['id']}  {s['updated_at']}  {s['title']}")
        return True
    if name == "/use":
        if rest.isdigit():
            args.session = int(rest)
            print(f"已切换到会话 {args.session}")
        else:
            print("用法：/use <会话ID>")
        return True
    if name == "/facts":
        facts = memory.all_facts()
        print("\n".join(f"  · {k} = {v}" for k, v in facts.items()) or "  （暂无长期记忆）")
        return True
    if name == "/remember":
        if "=" in rest:
            k, v = rest.split("=", 1)
            memory.remember(k.strip(), v.strip())
            print(f"已记住：{k.strip()} = {v.strip()}")
        else:
            print("用法：/remember 键=值")
        return True
    if name == "/forget":
        if rest:
            memory.forget(rest)
            print(f"已忘掉：{rest}")
        else:
            print("用法：/forget 键")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="小枢 · MCP 智能体命令行")
    parser.add_argument("--once", type=str, default="", help="执行单条指令后退出")
    parser.add_argument("--session", type=int, default=0, help="复用已有会话 ID")
    parser.add_argument("--voice", action="store_true", help="开启语音输入/输出")
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(run_cli(args)))
    except KeyboardInterrupt:
        print("\n已中断。")


if __name__ == "__main__":
    main()
