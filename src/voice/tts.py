"""文字转语音（TTS）：pyttsx3 离线朗读，Windows 走 SAPI5，无需联网。"""
from __future__ import annotations

from ..common.logging_setup import get_logger

_LOG = get_logger("voice.tts")

_MAX_CHARS = 300  # 太长会念得人烦躁，超出部分省略


class VoiceDependencyMissing(RuntimeError):
    pass


def speak(text: str, rate: int = 180) -> None:
    """朗读文本。任何失败都只记日志，不能因为音箱没插就把整个 Agent 搞崩。"""
    if not text:
        return
    body = text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "，后续内容已省略"
    try:
        import pyttsx3
    except ImportError:
        raise VoiceDependencyMissing(
            "缺少语音依赖，请执行：pip install -r requirements-voice.txt"
        )
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.say(body)
        engine.runAndWait()
    except Exception as exc:
        _LOG.warning("语音合成失败: %s", exc)
