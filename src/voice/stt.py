"""语音转文字（STT）：麦克风录音 -> faster-whisper 本地转写。

为什么选本地模型：不需要把音频传到第三方，也不依赖外网，课堂演示更稳。
首次运行会自动下载模型（base 约 75MB），之后走缓存。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..common.logging_setup import get_logger

_LOG = get_logger("voice.stt")


class VoiceDependencyMissing(RuntimeError):
    pass


_MODEL_CACHE: dict[str, object] = {}


def _load_whisper(model_size: str):
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise VoiceDependencyMissing(
            "缺少语音依赖，请执行：pip install -r requirements-voice.txt"
        ) from exc
    # int8 量化：CPU 上速度快 2~3 倍，精度损失对中文短句基本无感
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _MODEL_CACHE[model_size] = model
    return model


def record_wav(seconds: float = 5.0, samplerate: int = 16000) -> Path:
    """录音并保存为临时 wav 文件，返回路径。

    用标准库 wave 写文件，避免额外引入 scipy/soundfile。
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as exc:
        raise VoiceDependencyMissing(
            "缺少录音依赖，请执行：pip install -r requirements-voice.txt"
        ) from exc

    _LOG.info("录音 %.1f 秒…", seconds)
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    import wave
    path = Path(tempfile.mkdtemp(prefix="agent-stt-")) / "input.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(samplerate)
        wf.writeframes(np.asarray(audio, dtype="int16").tobytes())
    return path


def listen_once(seconds: float = 5.0, model_size: str = "base", language: str = "zh") -> str:
    """录一段音并转写成文字。失败时返回空串而不是抛异常，避免打断对话。"""
    try:
        import sounddevice as sd  # noqa: F401
        wav = record_wav(seconds=seconds)
        model = _load_whisper(model_size)
        segments, info = model.transcribe(str(wav), language=language, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()
        _LOG.info("识别结果: %s", text)
        return text
    except VoiceDependencyMissing:
        raise
    except Exception as exc:
        _LOG.warning("语音识别失败: %s", exc)
        return ""
