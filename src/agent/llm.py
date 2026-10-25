"""LLM 适配层：OpenAI 兼容接口 + Mock 规则模型（无密钥也能跑通全链路）。

为什么要 Mock：
作业验收 / 课堂演示时，同学可能没 API Key、网络也不通。Mock 让"MCP 握手 -> 工具调用 ->
结果整合 -> 多轮记忆"这条主干链路照常跑通，方便先把工程骨架验收掉，再换真模型。

注意：Mock 是规则匹配，只能覆盖演示脚本里的固定句式，**不能用来评价模型能力**。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..common.config import Config, load_config
from ..common.errors import FatalError, LlmUnavailable
from ..common.logging_setup import get_logger
from ..common.retry import retry_async

_LOG = get_logger("llm")


@dataclass
class AssistantTurn:
    """一次模型回复的标准化结果。"""
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    model: str = ""


# --------------------------------------------------------------------------
# OpenAI 兼容实现
# --------------------------------------------------------------------------

class OpenAILlm:
    """调用任何 OpenAI 兼容的 /v1/chat/completions（OpenAI / DeepSeek / 通义 / Ollama / vLLM）。"""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        try:
            from openai import AsyncOpenAI  # 延迟导入：mock 模式不装 openai 也能跑
        except ImportError as exc:  # pragma: no cover
            raise FatalError("未安装 openai 包，请执行 pip install -r requirements.txt") from exc

        if not self.cfg.has_api_key:
            raise FatalError(
                "未配置 LLM_API_KEY。请复制 .env.example 为 .env 并填写，"
                "或设置 LLM_MODE=mock 使用内置规则模型。"
            )

        self.model = self.cfg.llm_model
        self._client = AsyncOpenAI(
            base_url=self.cfg.llm_base_url,
            api_key=self.cfg.llm_api_key,
            timeout=float(self.cfg.llm_timeout_sec),
            max_retries=0,  # 重试由本项目自己控制，避免与 SDK 内置重试叠加成重试风暴
        )

    def _translate_error(self, exc: BaseException) -> BaseException:
        """把 SDK 异常映射成"可重试 / 致命"两类，决定要不要退避重来。"""
        name = type(exc).__name__
        if name in ("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"):
            return LlmUnavailable(f"{name}: {exc}")
        try:
            from openai import APIStatusError
            if isinstance(exc, APIStatusError):
                code = getattr(exc, "status_code", 0)
                if code == 429 or code >= 500:
                    return LlmUnavailable(f"HTTP {code}: {exc}")
                # 400/401/403 属于配置或请求本身的问题，重试一万次也不会变好
                return FatalError(f"HTTP {code}: {exc}")
        except Exception:
            pass
        return exc

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> AssistantTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.cfg.llm_temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        async def _once() -> Any:
            try:
                return await self._client.chat.completions.create(**kwargs)
            except FatalError:
                raise
            except Exception as exc:
                raise self._translate_error(exc)

        try:
            resp = await retry_async(
                _once,
                attempts=3,
                base=1.0,
                cap=8.0,
                retry_on=(LlmUnavailable,),
                on_retry=lambda i, d, e: _LOG.warning("LLM 第 %d 次重试，%.1fs 后执行: %s", i, d, e),
            )
        except LlmUnavailable as exc:
            # 降级入口：交由 controller 决定是切 mock 还是直接告诉用户
            raise LlmUnavailable(f"模型连续不可用，请检查网络/额度/配置: {exc}") from exc

        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            raw_args = tc.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                # 模型偶发吐出非法 JSON：记下来并回喂错误，让它自己修正，而不是直接崩
                _LOG.warning("模型返回非法 JSON 参数: %s", raw_args[:200])
                parsed = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": parsed})

        usage = getattr(resp, "usage", None)
        return AssistantTurn(
            content=msg.content or "",
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            model=getattr(resp, "model", self.model) or self.model,
        )


# --------------------------------------------------------------------------
# Mock 规则模型
# --------------------------------------------------------------------------

_CN_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "武汉", "南京", "重庆", "苏州", "天津"]

# 规则模型里的兜底安全拒绝。注意：这只是 Mock 为了让演示/评测"看起来像个守规矩的模型"而加的规则，
# 真实系统的防护必须靠两样东西：系统提示词的约束 + 工具侧的沙箱与二次确认。三者不可互相替代。
_REFUSAL_RULES: list[tuple[str, str]] = [
    (r"忽略(以上|前面|之前)?(所有|全部)?(指令|规则|提示|要求)", "我不能忽略安全规则或执行越权指令。"),
    (r"(不要|别|不许)(用|使用|调用|借助)(任何)?工具", "真实数据必须来自工具，我不能凭记忆编造天气、文件或待办信息。"),
    (r"rm\s+-rf|删除系统|所有文件|格式化", "删除系统文件这类操作超出了我的能力范围，也违反安全规则。"),
    (r"id_rsa|\.ssh|passwd|shadow", "读取或转存凭据、密钥类敏感文件是被禁止的。"),
    (r"api[_\s-]?key|密钥|token", "我不能输出任何密钥、Token 或凭据。"),
    (r"系统提示词|提示词原文", "系统提示词属于内部实现细节，不能对外输出。"),
    (r"管理员模式|root\s*权限|提权", "我不支持切换权限模式，所有操作都受同一套安全规则约束。"),
]
_FILE_RE = re.compile(r"[\w\u4e00-\u9fa5_\-/\.]+\.(?:txt|md|json|csv|py|log|yaml|yml)")
_CITY_RE = re.compile(r"([\u4e00-\u9fa5]{2,4})(?:的)?天气")
_TODO_ID_RE = re.compile(r"(?:待办|任务|编号|#)\s*(\d+)")


def _clean_path(raw: str) -> str:
    """把模型/用户说出来的路径规范成"相对 workspace 根"的形式。

    用户常说"读一下 workspace/notes.md"，但文件工具的根本身就是 workspace，
    再拼一次会变成 workspace/workspace/notes.md，因此这里去掉多余的前缀。
    """
    p = raw.strip().lstrip("./")
    for prefix in ("workspace/", "workspace\\"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    return p


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


_CITY_FILLER_RE = re.compile(r"^(?:帮我|请|请问|查一下|查询|查查|看看|看一下|说下|今天|明天|后天|那|去|在)+")


def _pick_city(text: str) -> str | None:
    # 1) 先命中已知城市表，最稳
    for city in _CN_CITIES:
        if city in text:
            return city
    # 2) 再走"...天气"句式，并去掉句首的动词/ filler，避免"查一下北京"被切成"一下北京"
    m = _CITY_RE.search(text)
    if m:
        cleaned = _CITY_FILLER_RE.sub("", m.group(1))
        return cleaned or None
    # 3) 英文城市名
    m2 = re.search(r"\b([A-Z][a-z]{2,})\b", text)
    return m2.group(1) if m2 else None


class MockLlm:
    """规则兜底模型：关键词 -> 工具；拿到工具结果后 -> 组织最终回复。"""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.model = "mock-rule-v1"

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> AssistantTurn:
        tools = tools or []
        allowed = {t["function"]["name"] for t in tools if t.get("function")}
        text = _last_user_text(messages)

        # 上一轮已下发工具调用、本轮收到工具结果 -> 进入"整合输出"分支
        if messages and messages[-1].get("role") == "tool":
            summary = self._summarize(messages)
            return AssistantTurn(
                content=summary,
                prompt_tokens=sum(len(str(m)) for m in messages) // 2,
                completion_tokens=len(summary) // 2,
                model=self.model,
            )

        refusal = self._refusal(text)
        if refusal:
            return AssistantTurn(content=refusal, prompt_tokens=len(text) // 2,
                                 completion_tokens=len(refusal) // 2, model=self.model)

        call = self._route(text, allowed)
        if call is None:
            reply = (
                "【Mock 模型】我没在你这句话里匹配到可用的工具指令。\n"
                "可以试试这些说法：\n"
                "  · 查一下北京的天气\n"
                "  · 添加待办：周五前完成作业提交\n"
                "  · 列出我的待办\n"
                "  · 看看 workspace 目录里有什么\n"
                "  · 现在几点了\n"
                "（设置 LLM_MODE=openai 并填好 API Key 后，就能用真正的自然语言对话了）"
            )
            return AssistantTurn(content=reply, prompt_tokens=len(text) // 2,
                                 completion_tokens=len(reply) // 2, model=self.model)

        return AssistantTurn(
            content="",
            tool_calls=[{"id": f"call_{abs(hash(text)) % 100000}", "name": call[0], "arguments": call[1]}],
            prompt_tokens=len(text) // 2,
            completion_tokens=40,
            finish_reason="tool_calls",
            model=self.model,
        )

    def _refusal(self, text: str) -> str | None:
        """命中安全规则时返回一句拒绝话术，否则返回 None。"""
        for pattern, reason in _REFUSAL_RULES:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return f"【Mock 模型·安全拒绝】{reason}如果你有其他需求，我可以继续帮忙。"
        return None

    def _route(self, text: str, allowed: set[str]) -> tuple[str, dict[str, Any]] | None:
        t = text.strip()

        # 写操作放在读操作之前匹配：避免出现"写到 x.md"被读分支误吞
        if ("写" in t or "保存" in t or "生成文件" in t or "导出" in t) and "fs_write_file" in allowed:
            m = _FILE_RE.search(t)
            if m:
                return "fs_write_file", {"path": _clean_path(m.group(0)), "content": f"# 由 Agent 写入\n\n来源指令：{t}"}

        if "天气" in t and "weather_get" in allowed:
            city = _pick_city(t)
            if not city:
                return None
            days = 3
            m = re.search(r"(\d)\s*天", t)
            if m:
                days = int(m.group(1))
            return "weather_get", {"city": city, "days": days}

        if ("待办" in t or "提醒我" in t or "记一下" in t or "任务" in t) and (
            "添加" in t or "新增" in t or "加" in t or "帮我记" in t or "提醒我" in t
        ) and "todo_add" in allowed:
            title = re.sub(r"^(帮我|请)?(添加|新增|加个|加|记一下|提醒我)(一条)?(待办|任务)?[：:，,]?\s*", "", t).strip()
            if not title:
                title = t
            priority = "high" if any(w in t for w in ("紧急", "重要", "马上")) else "medium"
            return "todo_add", {"title": title[:60], "priority": priority}

        if ("待办" in t or "清单" in t or "任务" in t) and ("列出" in t or "查看" in t or "看看" in t or "有哪些" in t) \
                and "todo_list" in allowed:
            status = "all" if "全部" in t or "所有" in t else "open"
            return "todo_list", {"status": status}

        if ("完成" in t or "做完了" in t or "搞定" in t) and "todo_done" in allowed:
            m = _TODO_ID_RE.search(t)
            if m:
                return "todo_done", {"id": int(m.group(1))}
            return None

        if ("删除" in t or "移除" in t) and ("待办" in t or "任务" in t) and "todo_remove" in allowed:
            m = _TODO_ID_RE.search(t)
            if m:
                return "todo_remove", {"id": int(m.group(1))}
            return None

        if ("目录" in t or "文件夹" in t or "有什么文件" in t or "列一下" in t) and "fs_list_dir" in allowed:
            return "fs_list_dir", {"path": ".", "recursive": False}

        if ("统计" in t or "分析" in t) and "fs_analyze_file" in allowed:
            m = _FILE_RE.search(t)
            if m:
                return "fs_analyze_file", {"path": _clean_path(m.group(0))}
            return None

        if ("读" in t or "打开" in t or "看一下文件" in t or "内容" in t) and "fs_read_file" in allowed:
            m = _FILE_RE.search(t)
            if m:
                return "fs_read_file", {"path": _clean_path(m.group(0))}
            return None

        if ("几点" in t or "现在时间" in t or "今天星期" in t or "日期" in t) and "time_now" in allowed:
            return "time_now", {}

        if ("算" in t or "计算" in t or "等于" in t) and "code_run_python" in allowed and self.cfg.allow_code_exec:
            expr = re.sub(r"^[^\d\(\[]*", "", t)
            if expr:
                return "code_run_python", {"code": expr[:200]}
            return None

        return None

    def _summarize(self, messages: list[dict[str, Any]]) -> str:
        """把"本轮"的工具结果拼成人话。真实模型会自己写，这里用固定模板兜底。"""
        # 只取最后一条 user 消息之后的工具结果，否则会把历史轮次的结果重复念一遍
        last_user_idx = 0
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                last_user_idx = i
        results = [m for m in messages[last_user_idx:] if m.get("role") == "tool"]
        parts: list[str] = []
        for m in results:
            raw = m.get("content", "")
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parts.append(str(raw)[:800])
                continue

            if "forecast" in data and "current" in data:
                cur = data["current"]
                lines = [f"{data.get('city')}当前{cur.get('weather')}，气温 {cur.get('temperature')}℃，"
                         f"体感 {cur.get('feels_like')}℃，湿度 {cur.get('humidity')}%。"]
                for f in data.get("forecast", [])[:3]:
                    lines.append(f"  · {f.get('date')}：{f.get('weather')}，"
                                 f"{f.get('temp_min')}~{f.get('temp_max')}℃，降水概率 {f.get('rain_prob')}%")
                parts.append("\n".join(lines))
            elif "items" in data:
                items = data["items"]
                if not items:
                    parts.append("当前没有符合条件的待办。")
                else:
                    head = f"共 {data.get('count')} 条待办："
                    body = "\n".join(
                        f"  · #{i['id']} [{i['priority']}] {i['title']}"
                        + (f"（截止 {i['due']}）" if i.get("due") else "")
                        for i in items
                    )
                    parts.append(head + "\n" + body)
            elif data.get("message"):
                parts.append(str(data["message"]))
            elif "entries" in data:
                entries = data["entries"][:20]
                parts.append("目录内容：\n" + "\n".join(
                    f"  · {e.get('type')}\t{e.get('path')}" for e in entries))
            elif "local_time" in data:
                parts.append(f"现在是 {data['local_time']}（{data.get('weekday')}）。")
            else:
                parts.append(json.dumps(data, ensure_ascii=False)[:800])

        return "【Mock 模型】已为你完成：\n" + "\n\n".join(parts)


def build_llm(cfg: Config | None = None):
    """工厂：按配置返回可用模型；openai 模式缺 Key 时自动降级到 mock 并明确告警。"""
    cfg = cfg or load_config()
    if cfg.llm_mode == "mock" or not cfg.has_api_key:
        if cfg.llm_mode == "openai":
            _LOG.warning("未检测到可用 API Key，自动降级为 Mock 规则模型（链路可跑，但非真实语义理解）")
        return MockLlm(cfg)
    return OpenAILlm(cfg)
