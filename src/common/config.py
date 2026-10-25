"""全局配置：全部从环境变量读取，禁止把密钥写进代码或日志（C6）。

加载优先级：进程环境变量 > .env 文件 > 内置默认值。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录：<root>/src/common/config.py -> parents[2] = <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv_once() -> None:
    """读取 .env（若存在）。python-dotenv 缺失时静默跳过，保证零配置也能启动。"""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_path(value: str | None, default: str) -> Path:
    raw = (value or default).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


@dataclass(frozen=True)
class Config:
    # ---- 大模型 ----
    llm_mode: str = "openai"            # openai | mock
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_timeout_sec: int = 60

    # ---- 有界自主预算 ----
    max_steps: int = 10
    max_wall_clock_sec: int = 120
    max_token_budget: int = 60000
    tool_timeout_sec: int = 20

    # ---- 路径与沙箱 ----
    workspace_root: Path = field(default_factory=lambda: PROJECT_ROOT / "workspace")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".data")

    # ---- 安全开关 ----
    require_confirm: bool = True
    allow_code_exec: bool = False
    allow_http_fetch: bool = True

    # ---- 日志 ----
    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.db"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def has_api_key(self) -> bool:
        return bool(self.llm_api_key and not self.llm_api_key.startswith("sk-xxxx"))


def load_config() -> Config:
    _load_dotenv_once()
    cfg = Config(
        llm_mode=(os.getenv("LLM_MODE") or "openai").strip().lower(),
        llm_base_url=(os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1").strip(),
        llm_api_key=(os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip(),
        llm_model=(os.getenv("LLM_MODEL") or "deepseek-chat").strip(),
        llm_temperature=_as_float(os.getenv("LLM_TEMPERATURE"), 0.2),
        llm_timeout_sec=_as_int(os.getenv("LLM_TIMEOUT_SEC"), 60),
        max_steps=_as_int(os.getenv("MAX_STEPS"), 10),
        max_wall_clock_sec=_as_int(os.getenv("MAX_WALL_CLOCK_SEC"), 120),
        max_token_budget=_as_int(os.getenv("MAX_TOKEN_BUDGET"), 60000),
        tool_timeout_sec=_as_int(os.getenv("TOOL_TIMEOUT_SEC"), 20),
        workspace_root=_as_path(os.getenv("WORKSPACE_ROOT"), "./workspace"),
        data_dir=_as_path(os.getenv("DATA_DIR"), "./.data"),
        require_confirm=_as_bool(os.getenv("REQUIRE_CONFIRM"), True),
        allow_code_exec=_as_bool(os.getenv("ALLOW_CODE_EXEC"), False),
        allow_http_fetch=_as_bool(os.getenv("ALLOW_HTTP_FETCH"), True),
        log_level=(os.getenv("LOG_LEVEL") or "INFO").strip().upper(),
    )
    # 目录必须存在，否则工具与数据库首次调用会失败
    cfg.workspace_root.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    return cfg
