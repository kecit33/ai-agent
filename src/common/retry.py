"""带指数退避 + 抖动的异步重试。

为什么自己写而不用 tenacity：
1. 只依赖标准库，作业环境少一个依赖就少一个装不上的风险；
2. 抖动（jitter）必须显式可见——它是避免多个请求同时重试打爆上游的关键，
   藏在第三方库里就讲不清了。
"""
from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    """第 attempt 次重试前的等待秒数（attempt 从 0 开始）。"""
    delay = min(cap, base * (2 ** attempt))
    return delay * (0.5 + random.random() / 2.0)  # 全抖动，避免重试风暴


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base: float = 0.5,
    cap: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_retry=None,
) -> T:
    """执行 fn，失败按指数退避重试。最后一次仍失败则抛出原异常。

    attempts=3 表示：最多尝试 3 次（2 次重试）。
    """
    last_exc: BaseException | None = None
    for i in range(attempts):
        try:
            return await fn()
        except retry_on as exc:  # noqa: B902
            last_exc = exc
            if i == attempts - 1:
                break
            delay = backoff_delay(i, base=base, cap=cap)
            if on_retry is not None:
                on_retry(i + 1, delay, exc)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
