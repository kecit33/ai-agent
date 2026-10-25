"""联网类工具：天气查询（外部 API）、通用 JSON 拉取、当前时间。

选 Open-Meteo 的原因：免费、无需 API Key，作业演示不用为申请密钥卡住。
[待核实: 接口字段以 https://open-meteo.com/en/docs 官方文档为准]
"""
from __future__ import annotations

import os
import time
from urllib.parse import urlparse

import httpx

from ..common.config import load_config
from ..common.errors import PermissionDenied, UserInputError

_CFG = load_config()
HTTP_TIMEOUT = 10.0
MAX_RESPONSE_CHARS = 2000

# 通用抓取的域名白名单（防 SSRF）。可通过环境变量 HTTP_ALLOWED_DOMAINS 追加，逗号分隔。
ALLOWED_DOMAINS: set[str] = {
    "api.open-meteo.com",      # 天气
    "wttr.in",                 # 天气（备用数据源）
    "api.github.com",          # 公开只读接口，演示用
    "raw.githubusercontent.com",
}
_extra = (os.getenv("HTTP_ALLOWED_DOMAINS") or "").strip()
if _extra:
    ALLOWED_DOMAINS.update(d.strip().lower() for d in _extra.split(",") if d.strip())


def _host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _domain_allowed(url: str) -> bool:
    """主机名精确匹配白名单（不做后缀通配，避免 evil-open-meteo.com 这类绕过）。"""
    host = _host_of(url)
    return host in ALLOWED_DOMAINS

# WMO weather code -> 中文描述（Open-Meteo 返回的 weather_code 遵循 WMO 4677 编码）
# [待核实: 完整映射表见 https://open-meteo.com/en/docs 的 "Weather variables" 章节]
WMO_CODE_CN: dict[int, str] = {
    0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴小冰雹", 99: "雷阵雨伴大冰雹",
}


def _describe(code: int | None) -> str:
    if code is None:
        return "未知"
    return WMO_CODE_CN.get(int(code), f"未知天气码 {code}")


async def _geocode(client: httpx.AsyncClient, city: str) -> tuple[float, float, str]:
    resp = await client.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh", "format": "json"},
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    if not results:
        raise UserInputError(f"没找到城市: {city}，请换成更标准的名称或加上国家/省份")
    first = results[0]
    return float(first["latitude"]), float(first["longitude"]), first.get("name", city)


async def weather_get(args: dict) -> dict:
    city = str(args["city"]).strip()
    if not city:
        raise UserInputError("city 不能为空")
    days = int(args.get("days", 3))
    days = max(1, min(days, 7))

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        lat, lon, name = await _geocode(client, city)
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": days,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    current = data.get("current", {})
    daily = data.get("daily", {})
    forecast = []
    dates = daily.get("time", [])
    for i, date in enumerate(dates):
        forecast.append({
            "date": date,
            "weather": _describe((daily.get("weather_code") or [None] * len(dates))[i]),
            "temp_max": (daily.get("temperature_2m_max") or [None] * len(dates))[i],
            "temp_min": (daily.get("temperature_2m_min") or [None] * len(dates))[i],
            "rain_prob": (daily.get("precipitation_probability_max") or [None] * len(dates))[i],
        })
    return {
        "city": name,
        "latitude": lat,
        "longitude": lon,
        "timezone": data.get("timezone"),
        "current": {
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "weather": _describe(current.get("weather_code")),
        },
        "forecast": forecast,
    }


async def http_get_json(args: dict) -> dict:
    """受限的 HTTP GET：https 强制 + 域名白名单 + 10s 超时 + 响应截断。

    白名单是本工具的核心防护：没有它，模型一旦被提示注入诱导，就能用服务器身份
    访问内网地址（SSRF）。默认只放行教学演示需要的公共接口。
    """
    if not _CFG.allow_http_fetch:
        raise PermissionDenied("已通过 ALLOW_HTTP_FETCH=false 关闭外网抓取")
    url = str(args["url"]).strip()
    if not url.startswith("https://"):
        raise PermissionDenied("只允许 https:// 开头的地址，拒绝明文 http 与 file:// 等协议")
    if not _domain_allowed(url):
        raise PermissionDenied(
            f"域名不在白名单内：{_host_of(url)}。允许：{', '.join(sorted(ALLOWED_DOMAINS))}"
        )

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        text = resp.text[:MAX_RESPONSE_CHARS]

    parsed: object
    try:
        import json as _json
        parsed = _json.loads(text)
        kind = "json"
    except Exception:
        parsed = text
        kind = "text"
    return {"url": url, "status_code": resp.status_code, "kind": kind, "body": parsed}


async def time_now(args: dict) -> dict:
    tz_name = args.get("timezone")
    now = time.time()
    local = time.localtime(now)
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][local.tm_wday]
    return {
        "unix": now,
        "local_time": time.strftime("%Y-%m-%d %H:%M:%S", local),
        "weekday": weekday_cn,
        "timezone_arg": tz_name,   # 注：本实现以运行机器本地时区为准，不做多时区换算
    }
