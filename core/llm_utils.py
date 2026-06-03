"""LLM 调用工具——自动重试 + 指数退避 + 降级策略 + 共享客户端"""

import time
import logging
from typing import TypeVar, Callable, Any

from openai import OpenAI

logger = logging.getLogger("eval-agent")

# ── 共享客户端池（单例，避免重复创建）──
_shared_clients: dict[str, OpenAI] = {}


def get_llm_client(provider: str = None) -> OpenAI:
    """获取共享的 OpenAI 客户端（单例模式）

    Args:
        provider: LLM 提供商，None 时使用 config 默认值

    Returns:
        OpenAI 客户端实例
    """
    from config import settings

    provider = provider or settings.llm_provider
    if provider not in _shared_clients:
        kwargs = {"api_key": settings.openai_api_key, "timeout": 300.0}
        if provider == "dashscope":
            kwargs["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            kwargs["base_url"] = settings.openai_api_base
        _shared_clients[provider] = OpenAI(**kwargs)
    return _shared_clients[provider]

T = TypeVar("T")
FALLBACK = TypeVar("FALLBACK")

# 可重试的异常类型
RETRIABLE_ERRORS = (
    ConnectionError, TimeoutError, OSError,
)

MAX_RETRIES = 3
BASE_DELAY = 2.0  # 秒


def safe_llm_call(
    fn: Callable[..., T],
    *args,
    max_retries: int = MAX_RETRIES,
    fallback: Any = None,
    label: str = "LLM call",
    **kwargs,
) -> T | Any:
    """安全调用 LLM，支持自动重试 + 指数退避 + 降级

    Args:
        fn: 要调用的函数（通常是一个 lambda 包裹的 LLM API 调用）
        max_retries: 最大重试次数
        fallback: 所有重试失败后的降级返回值
        label: 日志标签

    Returns:
        函数返回值，或降级值
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"[{label}] 第 {attempt+1}/{max_retries+1} 次调用失败: {e}，"
                    f"{delay:.0f}s 后重试..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"[{label}] {max_retries+1} 次调用全部失败，返回降级值: {e}"
                )

    if fallback is not None:
        return fallback

    # 无降级值时重新抛出最后一次异常
    raise last_error
